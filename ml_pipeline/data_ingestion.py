"""Data ingestion module for loading and validating raw data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handle data loading and initial validation."""

    def __init__(self, data_path: Path, target_column: str = "Churn"):
        """
        Initialize data ingestion.

        Args:
            data_path: Path to the raw data file
            target_column: Name of the target column
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.raw_data: Optional[pd.DataFrame] = None
        self.data_info: Dict[str, Any] = {}

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load data from CSV file.

        Returns:
            DataFrame with raw data or None if error occurs
        """
        try:
            if not self.data_path.exists():
                logger.error(f"Data file not found at {self.data_path}")
                return None

            self.raw_data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")

            # Store data info
            self.data_info = {
                "shape": self.raw_data.shape,
                "columns": self.raw_data.columns.tolist(),
                "memory_usage_mb": self.raw_data.memory_usage(deep=True).sum()
                / 1024**2,
                "null_counts": self.raw_data.isnull().sum().to_dict(),
            }

            logger.info(f"Data info: {self.data_info}")
            return self.raw_data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            return None

    def validate_data(self) -> bool:
        """
        Validate that data has required structure.

        Returns:
            True if data is valid, False otherwise
        """
        if self.raw_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False

        try:
            # Check if target column exists
            if self.target_column not in self.raw_data.columns:
                logger.error(f"Target column '{self.target_column}' not found in data")
                return False

            # Check for empty dataframe
            if self.raw_data.empty:
                logger.error("DataFrame is empty")
                return False

            # Check minimum rows
            if len(self.raw_data) < 10:
                logger.warning(f"Very small dataset: {len(self.raw_data)} rows")

            logger.info("Data validation passed")
            return True

        except Exception as e:
            logger.error(f"Error validating data: {str(e)}", exc_info=True)
            return False

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.

        Returns:
            Dictionary with data information
        """
        return self.data_info.copy() if self.data_info else {}
