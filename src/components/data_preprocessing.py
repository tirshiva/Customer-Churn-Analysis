"""Data preprocessing module for cleaning and transforming data."""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from ml_pipeline.dimension_reduction import DimensionReducer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing and feature engineering."""
    
    def __init__(
        self,
        target_column: str = "Churn",
        dimension_reduction: Optional[DimensionReducer] = None
    ):
        """
        Initialize data preprocessor.
        
        Args:
            target_column: Name of the target column
            dimension_reduction: Optional dimension reducer instance
        """
        self.target_column = target_column
        self.dimension_reducer = dimension_reduction
        self.processed_data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.preprocessing_info: Dict[str, Any] = {}
    
    def preprocess(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Preprocess the data.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame or None if error occurs
        """
        try:
            if data is None or data.empty:
                logger.error("Input data is None or empty")
                return None
            
            # Create a copy to avoid modifying original
            processed_data = data.copy()
            
            # Step 1: Drop irrelevant columns
            processed_data = self._drop_irrelevant_columns(processed_data)
            
            # Step 2: Handle missing values
            processed_data = self._handle_missing_values(processed_data)
            
            # Step 3: Encode categorical variables
            processed_data = self._encode_categorical_variables(processed_data)
            
            # Step 4: Handle target variable
            y = None
            if self.target_column in processed_data.columns:
                y = processed_data[self.target_column].map({'Yes': 1, 'No': 0})
                processed_data[self.target_column] = y
                logger.info("Converted target column to numeric")
            
            # Step 5: Apply dimension reduction (if configured)
            if self.dimension_reducer is not None:
                logger.info("Applying dimension reduction...")
                # Separate features and target
                X_features = processed_data.drop(self.target_column, axis=1) if self.target_column in processed_data.columns else processed_data
                
                # Apply dimension reduction
                X_reduced = self.dimension_reducer.fit_transform(X_features, y)
                
                if X_reduced is not None:
                    # Combine reduced features with target
                    if self.target_column in processed_data.columns:
                        processed_data = pd.concat([X_reduced, processed_data[[self.target_column]]], axis=1)
                    else:
                        processed_data = X_reduced
                    logger.info(f"Dimension reduction applied. New shape: {processed_data.shape}")
                else:
                    logger.warning("Dimension reduction failed, using all features")
            
            # Store feature columns (excluding target)
            if self.target_column in processed_data.columns:
                self.feature_columns = [
                    col for col in processed_data.columns 
                    if col != self.target_column
                ]
            else:
                self.feature_columns = processed_data.columns.tolist()
            
            self.processed_data = processed_data
            
            # Get reduction info if available
            reduction_info = {}
            if self.dimension_reducer is not None:
                reduction_info = self.dimension_reducer.get_reduction_info()
            
            self.preprocessing_info = {
                "original_shape": data.shape,
                "processed_shape": processed_data.shape,
                "feature_count": len(self.feature_columns),
                "feature_names": self.feature_columns,
                "dimension_reduction": reduction_info
            }
            
            logger.info(f"Preprocessing completed. Shape: {processed_data.shape}")
            logger.info(f"Number of features: {len(self.feature_columns)}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
            return None
    
    def _drop_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are not useful for prediction."""
        columns_to_drop = ['customerID']
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
                logger.info(f"Dropped column: {col}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        try:
            # Convert TotalCharges to numeric
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Identify numerical and categorical columns
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_cols = [
                col for col in df.columns 
                if col not in numerical_cols and col != self.target_column
            ]
            
            # Fill numerical columns with median
            for col in numerical_cols:
                if col in df.columns:
                    median_val = df[col].median()
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        df[col].fillna(median_val, inplace=True)
                        logger.info(f"Filled {missing_count} missing values in {col} with median: {median_val}")
            
            # Fill categorical columns with mode
            for col in categorical_cols:
                if col in df.columns:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        df[col].fillna(mode_val, inplace=True)
                        logger.info(f"Filled {missing_count} missing values in {col} with mode: {mode_val}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}", exc_info=True)
            return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using one-hot encoding."""
        try:
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_cols = [
                col for col in df.columns 
                if col not in numerical_cols and col != self.target_column
            ]
            
            # Apply one-hot encoding
            if categorical_cols:
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
                logger.info(f"One-hot encoded {len(categorical_cols)} categorical columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error encoding categorical variables: {str(e)}", exc_info=True)
            return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns.copy()
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get preprocessing information."""
        return self.preprocessing_info.copy()
    
    def get_dimension_reducer(self) -> Optional[DimensionReducer]:
        """Get the dimension reducer instance."""
        return self.dimension_reducer

