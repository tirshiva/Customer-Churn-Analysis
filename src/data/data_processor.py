import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = None

    def load_data(self):
        """Load the raw data from CSV file."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully with shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

    def preprocess_data(self):
        """Preprocess the data."""
        if self.data is None:
            logger.error("Please load data first using load_data()")
            return None

        try:
            # Create a copy of the data
            processed_data = self.data.copy()

            # Drop customerID as it's not relevant for prediction
            if 'customerID' in processed_data.columns:
                processed_data = processed_data.drop('customerID', axis=1)
                logger.info("Dropped customerID column")

            # Handle missing values
            processed_data = self._handle_missing_values(processed_data)
            logger.info("Handled missing values")

            # Convert categorical variables
            processed_data = self._encode_categorical_variables(processed_data)
            logger.info("Encoded categorical variables")

            # Log the final columns
            logger.info(f"Final columns: {processed_data.columns.tolist()}")
            logger.info(f"Final shape: {processed_data.shape}")

            return processed_data
        except Exception as e:
            logger.error(f"Error in preprocessing data: {str(e)}")
            return None

    def _handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        try:
            # Convert TotalCharges to numeric, replacing any non-numeric values with NaN
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Identify numerical and categorical columns
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_cols = [col for col in df.columns if col not in numerical_cols and col != 'Churn']

            # Fill numerical columns with median
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

            # Fill categorical columns with mode
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])

            return df
        except Exception as e:
            logger.error(f"Error in handling missing values: {str(e)}")
            return df

    def _encode_categorical_variables(self, df):
        """Encode categorical variables."""
        try:
            # Define categorical columns (excluding Churn and numerical columns)
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_cols = [col for col in df.columns if col not in numerical_cols and col != 'Churn']

            # Apply one-hot encoding to categorical columns
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

            # Convert 'Churn' to numeric if it exists
            if 'Churn' in df.columns:
                df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
                logger.info("Converted Churn column to numeric")

            return df
        except Exception as e:
            logger.error(f"Error in encoding categorical variables: {str(e)}")
            return df

    def save_processed_data(self, processed_data, output_path):
        """Save the processed data to a CSV file."""
        try:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            processed_data.to_csv(output_path, index=False)
            logger.info(f"Processed data saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")

if __name__ == "__main__":
    # Get the current file's directory
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent

    # Set up paths
    input_path = project_root / "data" / "raw" / "data.csv"
    output_path = project_root / "data" / "processed" / "processed_data.csv"

    logger.info(f"Processing data from {input_path}")
    logger.info(f"Output will be saved to {output_path}")

    # Initialize and run data processor
    data_processor = DataProcessor(input_path)
    data = data_processor.load_data()
    if data is not None:
        processed_data = data_processor.preprocess_data()
        if processed_data is not None:
            data_processor.save_processed_data(processed_data, output_path) 