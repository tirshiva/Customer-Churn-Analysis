"""Data processing service."""

import pandas as pd
import logging
from typing import Dict, Any, Optional

from app.core.config import settings
from ml_pipeline.dimension_reduction import DimensionReducer

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handle data preprocessing for predictions."""

    def __init__(self):
        """Initialize the data processor."""
        self.categorical_columns = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
        self.numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
        self.dimension_reducer: Optional[DimensionReducer] = None
        self._load_reducer()

    def _load_reducer(self) -> None:
        """Load dimension reducer if it exists."""
        try:
            if settings.REDUCER_PATH.exists():
                self.dimension_reducer = DimensionReducer.load(settings.REDUCER_PATH)
                if self.dimension_reducer:
                    logger.info(f"Dimension reducer loaded from {settings.REDUCER_PATH}")
                    logger.info(f"  Method: {self.dimension_reducer.method}")
                    if self.dimension_reducer.selected_features:
                        logger.info(
                            f"  Selected features: {len(self.dimension_reducer.selected_features)}"
                        )
            else:
                logger.info("No dimension reducer found. Using all features.")
        except Exception as e:
            logger.warning(f"Could not load dimension reducer: {str(e)}")
            self.dimension_reducer = None

    def preprocess_input(self, input_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Preprocess input data for prediction.

        Args:
            input_data: Dictionary containing customer features

        Returns:
            Preprocessed DataFrame or None if error occurs
        """
        try:
            # Create DataFrame with single row
            df = pd.DataFrame([input_data])

            # Convert TotalCharges to numeric
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

            # Fill missing values in numerical columns
            for col in self.numerical_columns:
                if col in df.columns:
                    if df[col].isna().any():
                        # Use median if available, otherwise use 0
                        median_val = df[col].median() if not df[col].isna().all() else 0
                        df[col].fillna(median_val, inplace=True)

            # One-hot encode categorical variables
            for col in self.categorical_columns:
                if col in df.columns:
                    # Create dummy variables
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(col, axis=1, inplace=True)

            logger.info(f"Successfully preprocessed input data. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error preprocessing input data: {str(e)}", exc_info=True)
            return None

    def align_features(
        self, processed_data: pd.DataFrame, model_features: list
    ) -> Optional[pd.DataFrame]:
        """
        Align processed data features with model features.

        Args:
            processed_data: Preprocessed DataFrame
            model_features: List of feature names expected by the model

        Returns:
            Aligned DataFrame or None if error occurs
        """
        try:
            # Apply dimension reduction if available
            if self.dimension_reducer is not None and self.dimension_reducer.reducer is not None:
                logger.info("Applying dimension reduction to input data...")
                # First align with original features
                original_features = self.dimension_reducer.original_feature_names
                if original_features:
                    # Ensure all original features are present
                    for feature in original_features:
                        if feature not in processed_data.columns:
                            processed_data[feature] = 0
                            logger.debug(
                                f"Added missing original feature '{feature}' with default value 0"
                            )
                    # Select only original features in correct order
                    processed_data = processed_data[original_features]

                # Apply dimension reduction
                reduced_data = self.dimension_reducer.transform(processed_data)
                if reduced_data is not None:
                    processed_data = reduced_data
                    logger.info(f"Dimension reduction applied. New shape: {processed_data.shape}")
                else:
                    logger.warning("Dimension reduction failed, using original features")

            # Ensure all model features are present
            for feature in model_features:
                if feature not in processed_data.columns:
                    processed_data[feature] = 0
                    logger.warning(f"Missing feature '{feature}' added with default value 0")

            # Select only the features used by the model
            aligned_data = processed_data[model_features]

            # Ensure correct order
            aligned_data = aligned_data[model_features]

            logger.info(f"Features aligned. Final shape: {aligned_data.shape}")
            return aligned_data

        except Exception as e:
            logger.error(f"Error aligning features: {str(e)}", exc_info=True)
            return None
