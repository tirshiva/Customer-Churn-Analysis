import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load the processed data."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully with shape: {self.data.shape}")
            logger.info(f"Columns in data: {self.data.columns.tolist()}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

    def prepare_data(self):
        """Prepare data for training."""
        if self.data is None:
            logger.error("Please load data first using load_data()")
            return None

        try:
            # Separate features and target
            if 'Churn' not in self.data.columns:
                logger.error("'Churn' column not found in data")
                return None

            X = self.data.drop('Churn', axis=1)
            y = self.data['Churn']

            # Log feature names
            logger.info(f"Feature names: {X.columns.tolist()}")
            logger.info(f"Number of features: {len(X.columns)}")

            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            logger.info(f"Training set shape: {self.X_train.shape}")
            logger.info(f"Testing set shape: {self.X_test.shape}")
            return True
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None

    def train_model(self):
        """Train the model."""
        if self.X_train is None or self.y_train is None:
            logger.error("Please prepare data first using prepare_data()")
            return None

        try:
            # Initialize and train the model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            self.model.fit(self.X_train, self.y_train)
            logger.info("Model trained successfully")
            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None

    def evaluate_model(self):
        """Evaluate the model performance."""
        if self.model is None:
            logger.error("Please train the model first using train_model()")
            return None

        try:
            # Make predictions
            y_pred = self.model.predict(self.X_test)

            # Calculate and log metrics
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(self.y_test, y_pred))

            # Calculate and log confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            logger.info("\nConfusion Matrix:")
            logger.info("\n" + str(cm))

            # Calculate and log accuracy
            accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
            logger.info(f"\nAccuracy: {accuracy:.4f}")

            return True
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return None

    def save_model(self, model_path):
        """Save the trained model."""
        if self.model is None:
            logger.error("Please train the model first using train_model()")
            return None

        try:
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved successfully to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        # Get the current file's directory
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent

        # Set up paths
        input_path = project_root / "data" / "processed" / "processed_data.csv"
        model_path = project_root / "models" / "random_forest_model.joblib"

        logger.info(f"Loading data from {input_path}")
        logger.info(f"Model will be saved to {model_path}")

        # Initialize trainer
        trainer = ModelTrainer(input_path)

        # Run the training pipeline
        if trainer.load_data() is not None:
            if trainer.prepare_data():
                if trainer.train_model():
                    trainer.evaluate_model()
                    trainer.save_model(model_path)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}") 