"""Dimension reduction module for feature selection and dimensionality reduction."""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Literal
import logging
import joblib
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    SelectFromModel,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DimensionReducer:
    """Handle dimension reduction using various techniques."""

    def __init__(
        self,
        method: Literal[
            "pca", "selectkbest", "mutual_info", "select_from_model", "rfe", "none"
        ] = "select_from_model",
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        k_best: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize dimension reducer.

        Args:
            method: Dimension reduction method to use
                - "pca": Principal Component Analysis
                - "selectkbest": Select K best features using f_classif
                - "mutual_info": Select features using mutual information
                - "select_from_model": Select features using model importance
                - "rfe": Recursive Feature Elimination
                - "none": No dimension reduction
            n_components: Number of components/features to keep (for PCA)
            variance_threshold: Variance to retain (for PCA, 0-1)
            k_best: Number of best features to select (for selectkbest/mutual_info)
            random_state: Random state for reproducibility
        """
        self.method = method
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.k_best = k_best
        self.random_state = random_state

        # Reducer and scaler objects
        self.reducer: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.selected_features: Optional[List[str]] = None
        self.original_feature_names: Optional[List[str]] = None
        self.reduction_info: Dict[str, Any] = {}

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fit the reducer and transform the data.

        Args:
            X: Feature DataFrame
            y: Target Series (required for supervised methods)

        Returns:
            Transformed DataFrame or None if error
        """
        try:
            if X is None or X.empty:
                logger.error("Input data is None or empty")
                return None

            if self.method == "none":
                logger.info("No dimension reduction applied")
                self.selected_features = X.columns.tolist()
                self.original_feature_names = X.columns.tolist()
                return X

            # Store original feature names
            self.original_feature_names = X.columns.tolist()

            # Scale data for PCA
            if self.method == "pca":
                self.scaler = StandardScaler()
                X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X), columns=X.columns, index=X.index
                )
            else:
                X_scaled = X

            # Apply dimension reduction
            if self.method == "pca":
                result = self._apply_pca(X_scaled, y)
            elif self.method == "selectkbest":
                result = self._apply_selectkbest(X_scaled, y)
            elif self.method == "mutual_info":
                result = self._apply_mutual_info(X_scaled, y)
            elif self.method == "select_from_model":
                result = self._apply_select_from_model(X_scaled, y)
            elif self.method == "rfe":
                result = self._apply_rfe(X_scaled, y)
            else:
                logger.error(f"Unknown method: {self.method}")
                return None

            if result is None:
                return None

            # Store reduction info
            original_features = len(X.columns)
            reduced_features = result.shape[1]
            reduction_ratio = (1 - reduced_features / original_features) * 100

            self.reduction_info = {
                "method": self.method,
                "original_features": original_features,
                "reduced_features": reduced_features,
                "reduction_ratio": round(reduction_ratio, 2),
                "selected_features": self.selected_features,
            }

            logger.info("Dimension reduction completed:")
            logger.info(f"  Method: {self.method}")
            logger.info(f"  Original features: {original_features}")
            logger.info(f"  Reduced features: {reduced_features}")
            logger.info(f"  Reduction: {reduction_ratio:.2f}%")

            return result

        except Exception as e:
            logger.error(f"Error in dimension reduction: {str(e)}", exc_info=True)
            return None

    def transform(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Transform new data using fitted reducer.

        Args:
            X: Feature DataFrame

        Returns:
            Transformed DataFrame or None if error
        """
        try:
            if self.reducer is None:
                logger.error("Reducer not fitted. Call fit_transform() first.")
                return None

            if self.method == "none":
                return X

            # Scale if PCA was used
            if self.method == "pca" and self.scaler is not None:
                X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
            else:
                X_scaled = X

            # Transform
            if self.method == "pca":
                transformed = self.reducer.transform(X_scaled)
                # Create DataFrame with component names
                component_names = [f"PC{i+1}" for i in range(transformed.shape[1])]
                result = pd.DataFrame(transformed, columns=component_names, index=X.index)
            else:
                # For feature selection methods, just select features
                if self.selected_features:
                    result = X_scaled[self.selected_features].copy()
                else:
                    logger.error("No selected features available")
                    return None

            return result

        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}", exc_info=True)
            return None

    def _apply_pca(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Optional[pd.DataFrame]:
        """Apply PCA dimension reduction."""
        try:
            # Determine number of components
            if self.n_components is None:
                # Use variance threshold to determine components
                pca_temp = PCA()
                pca_temp.fit(X)
                cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= self.variance_threshold) + 1
                variance_pct = self.variance_threshold * 100
                logger.info(f"PCA: {n_components} components explain {variance_pct}% variance")
            else:
                n_components = min(self.n_components, X.shape[1])

            # Fit PCA
            self.reducer = PCA(n_components=n_components, random_state=self.random_state)
            transformed = self.reducer.fit_transform(X)

            # Create DataFrame
            component_names = [f"PC{i+1}" for i in range(transformed.shape[1])]
            result = pd.DataFrame(transformed, columns=component_names, index=X.index)

            # Store selected features (PC names)
            self.selected_features = component_names

            # Log explained variance
            explained_variance = np.sum(self.reducer.explained_variance_ratio_)
            logger.info(f"PCA explained variance: {explained_variance:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error applying PCA: {str(e)}", exc_info=True)
            return None

    def _apply_selectkbest(self, X: pd.DataFrame, y: pd.Series) -> Optional[pd.DataFrame]:
        """Apply SelectKBest feature selection."""
        try:
            if y is None:
                logger.error("Target variable required for SelectKBest")
                return None

            # Determine k
            if self.k_best is None:
                # Use default: keep top 50% or max 30 features
                k_best = min(max(10, X.shape[1] // 2), 30)
            else:
                k_best = min(self.k_best, X.shape[1])

            # Fit SelectKBest
            self.reducer = SelectKBest(score_func=f_classif, k=k_best)
            transformed = self.reducer.fit_transform(X, y)

            # Get selected feature names
            selected_indices = self.reducer.get_support(indices=True)
            self.selected_features = [X.columns[i] for i in selected_indices]

            # Create DataFrame
            result = pd.DataFrame(transformed, columns=self.selected_features, index=X.index)

            logger.info(f"SelectKBest: Selected {len(self.selected_features)} features")

            return result

        except Exception as e:
            logger.error(f"Error applying SelectKBest: {str(e)}", exc_info=True)
            return None

    def _apply_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> Optional[pd.DataFrame]:
        """Apply mutual information feature selection."""
        try:
            if y is None:
                logger.error("Target variable required for mutual information")
                return None

            # Determine k
            if self.k_best is None:
                k_best = min(max(10, X.shape[1] // 2), 30)
            else:
                k_best = min(self.k_best, X.shape[1])

            # Fit SelectKBest with mutual information
            self.reducer = SelectKBest(score_func=mutual_info_classif, k=k_best)
            transformed = self.reducer.fit_transform(X, y)

            # Get selected feature names
            selected_indices = self.reducer.get_support(indices=True)
            self.selected_features = [X.columns[i] for i in selected_indices]

            # Create DataFrame
            result = pd.DataFrame(transformed, columns=self.selected_features, index=X.index)

            logger.info(f"Mutual Information: Selected {len(self.selected_features)} features")

            return result

        except Exception as e:
            logger.error(f"Error applying mutual information: {str(e)}", exc_info=True)
            return None

    def _apply_select_from_model(self, X: pd.DataFrame, y: pd.Series) -> Optional[pd.DataFrame]:
        """Apply feature selection using model importance."""
        try:
            if y is None:
                logger.error("Target variable required for SelectFromModel")
                return None

            # Train a base model to get feature importance
            base_model = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )
            base_model.fit(X, y)

            # Determine threshold (median importance)
            importances = base_model.feature_importances_
            threshold = np.median(importances)

            # Fit SelectFromModel
            self.reducer = SelectFromModel(base_model, threshold=threshold, prefit=True)
            transformed = self.reducer.transform(X)

            # Get selected feature names
            selected_indices = self.reducer.get_support(indices=True)
            self.selected_features = [X.columns[i] for i in selected_indices]

            # Create DataFrame
            result = pd.DataFrame(transformed, columns=self.selected_features, index=X.index)

            logger.info(f"SelectFromModel: Selected {len(self.selected_features)} features")
            logger.info(f"  Threshold: {threshold:.6f}")

            return result

        except Exception as e:
            logger.error(f"Error applying SelectFromModel: {str(e)}", exc_info=True)
            return None

    def _apply_rfe(self, X: pd.DataFrame, y: pd.Series) -> Optional[pd.DataFrame]:
        """Apply Recursive Feature Elimination."""
        try:
            if y is None:
                logger.error("Target variable required for RFE")
                return None

            # Determine n_features_to_select
            if self.k_best is None:
                n_features = min(max(10, X.shape[1] // 2), 30)
            else:
                n_features = min(self.k_best, X.shape[1])

            # Create base estimator
            estimator = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )

            # Fit RFE
            self.reducer = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            transformed = self.reducer.fit_transform(X, y)

            # Get selected feature names
            selected_indices = self.reducer.get_support(indices=True)
            self.selected_features = [X.columns[i] for i in selected_indices]

            # Create DataFrame
            result = pd.DataFrame(transformed, columns=self.selected_features, index=X.index)

            logger.info(f"RFE: Selected {len(self.selected_features)} features")

            return result

        except Exception as e:
            logger.error(f"Error applying RFE: {str(e)}", exc_info=True)
            return None

    def get_selected_features(self) -> Optional[List[str]]:
        """Get list of selected feature names."""
        return self.selected_features.copy() if self.selected_features else None

    def get_reduction_info(self) -> Dict[str, Any]:
        """Get reduction information."""
        return self.reduction_info.copy()

    def save(self, filepath: Path) -> bool:
        """
        Save the reducer to disk.

        Args:
            filepath: Path to save the reducer

        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save reducer, scaler, and metadata
            save_dict = {
                "reducer": self.reducer,
                "scaler": self.scaler,
                "selected_features": self.selected_features,
                "original_feature_names": self.original_feature_names,
                "method": self.method,
                "reduction_info": self.reduction_info,
            }

            joblib.dump(save_dict, filepath)
            logger.info(f"Dimension reducer saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving reducer: {str(e)}", exc_info=True)
            return False

    @classmethod
    def load(cls, filepath: Path) -> Optional["DimensionReducer"]:
        """
        Load a saved reducer from disk.

        Args:
            filepath: Path to the saved reducer

        Returns:
            DimensionReducer instance or None if error
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.error(f"Reducer file not found: {filepath}")
                return None

            save_dict = joblib.load(filepath)

            # Create instance
            instance = cls(method=save_dict.get("method", "none"))
            instance.reducer = save_dict.get("reducer")
            instance.scaler = save_dict.get("scaler")
            instance.selected_features = save_dict.get("selected_features")
            instance.original_feature_names = save_dict.get("original_feature_names")
            instance.reduction_info = save_dict.get("reduction_info", {})

            logger.info(f"Dimension reducer loaded from {filepath}")
            return instance

        except Exception as e:
            logger.error(f"Error loading reducer: {str(e)}", exc_info=True)
            return None
