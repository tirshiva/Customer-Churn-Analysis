"""Complete ML pipeline orchestrator."""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import joblib

from ml_pipeline.data_ingestion import DataIngestion
from ml_pipeline.data_preprocessing import DataPreprocessor
from ml_pipeline.model_trainer import ModelTrainer
from ml_pipeline.model_evaluator import ModelEvaluator
from ml_pipeline.dimension_reduction import DimensionReducer

logger = logging.getLogger(__name__)


class MLPipeline:
    """Orchestrate the complete ML pipeline."""

    def __init__(
        self,
        data_path: Path,
        model_output_path: Path,
        evaluation_output_path: Optional[Path] = None,
        target_column: str = "Churn",
        model_type: str = "random_forest",
        random_state: int = 42,
        dimension_reduction_method: str = "select_from_model",
        dimension_reduction_k: Optional[int] = None,
    ):
        """
        Initialize ML pipeline.

        Args:
            data_path: Path to raw data file
            model_output_path: Path to save trained model
            evaluation_output_path: Path to save evaluation report
            target_column: Name of target column
            model_type: Type of model to train
            random_state: Random state for reproducibility
            dimension_reduction_method: Method for dimension reduction
            dimension_reduction_k: Number of features/components to keep
        """
        self.data_path = Path(data_path)
        self.model_output_path = Path(model_output_path)
        self.evaluation_output_path = (
            evaluation_output_path or self.model_output_path.parent / "evaluation_report.json"
        )
        self.reducer_output_path = self.model_output_path.parent / "dimension_reducer.joblib"
        self.target_column = target_column
        self.model_type = model_type
        self.random_state = random_state

        # Initialize dimension reducer
        self.dimension_reducer = DimensionReducer(
            method=dimension_reduction_method,
            k_best=dimension_reduction_k,
            random_state=random_state,
        )

        # Initialize components
        self.data_ingestion = DataIngestion(self.data_path, target_column)
        self.data_preprocessor = DataPreprocessor(target_column, self.dimension_reducer)
        self.model_trainer = ModelTrainer(model_type, random_state)
        self.model_evaluator = ModelEvaluator()

        # Pipeline state
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.trained_model: Optional[Any] = None
        self.pipeline_results: Dict[str, Any] = {}

    def run(self) -> bool:
        """
        Run the complete ML pipeline.

        Returns:
            True if pipeline completed successfully, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING ML PIPELINE")
            logger.info("=" * 60)

            # Step 1: Data Ingestion
            logger.info("\n[1/5] Data Ingestion")
            self.raw_data = self.data_ingestion.load_data()
            if self.raw_data is None:
                logger.error("Failed to load data")
                return False

            if not self.data_ingestion.validate_data():
                logger.error("Data validation failed")
                return False

            # Step 2: Data Preprocessing
            logger.info("\n[2/5] Data Preprocessing")
            self.processed_data = self.data_preprocessor.preprocess(self.raw_data)
            if self.processed_data is None:
                logger.error("Data preprocessing failed")
                return False

            # Step 3: Prepare Training Data
            logger.info("\n[3/5] Preparing Training Data")
            if not self.model_trainer.prepare_data(self.processed_data, self.target_column):
                logger.error("Failed to prepare training data")
                return False

            # Step 4: Train Model
            logger.info("\n[4/5] Training Model")
            if not self.model_trainer.train():
                logger.error("Model training failed")
                return False

            # Cross-validation
            cv_results = self.model_trainer.cross_validate()

            # Step 5: Evaluate Model
            logger.info("\n[5/5] Evaluating Model")
            evaluation_results = self.model_evaluator.evaluate(
                self.model_trainer.get_model(),
                self.model_trainer.X_test,
                self.model_trainer.y_test,
            )

            # Get feature importance
            feature_names = self.data_preprocessor.get_feature_columns()
            feature_importance = self.model_evaluator.get_feature_importance(
                self.model_trainer.get_model(), feature_names
            )

            # Store trained model
            self.trained_model = self.model_trainer.get_model()

            # Compile pipeline results
            self.pipeline_results = {
                "data_info": self.data_ingestion.get_data_info(),
                "preprocessing_info": self.data_preprocessor.get_preprocessing_info(),
                "training_info": self.model_trainer.get_training_info(),
                "cross_validation": cv_results,
                "evaluation": evaluation_results,
                "feature_importance": feature_importance,
            }

            # Save model
            logger.info(f"\nSaving model to {self.model_output_path}")
            self.save_model()

            # Save dimension reducer
            if self.dimension_reducer is not None and self.dimension_reducer.reducer is not None:
                logger.info(f"Saving dimension reducer to {self.reducer_output_path}")
                self.dimension_reducer.save(self.reducer_output_path)

            # Save evaluation report
            logger.info(f"Saving evaluation report to {self.evaluation_output_path}")
            self.model_evaluator.save_evaluation_report(self.evaluation_output_path)

            logger.info("\n" + "=" * 60)
            logger.info("ML PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False

    def save_model(self) -> bool:
        """
        Save the trained model.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.trained_model is None:
                logger.error("No model to save")
                return False

            self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.trained_model, self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            return False

    def get_results(self) -> Dict[str, Any]:
        """Get pipeline results."""
        return self.pipeline_results.copy()
