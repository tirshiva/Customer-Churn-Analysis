"""Advanced ML pipeline with multiple models and GridSearchCV."""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import joblib
import json

from ml_pipeline.data_ingestion import DataIngestion
from ml_pipeline.data_preprocessing import DataPreprocessor
from ml_pipeline.model_trainer import AdvancedModelTrainer
from ml_pipeline.model_evaluator import ModelEvaluator
from ml_pipeline.dimension_reduction import DimensionReducer
from ml_pipeline.model_visualizer import ModelVisualizer

logger = logging.getLogger(__name__)


class AdvancedMLPipeline:
    """Advanced ML pipeline with multiple models and hyperparameter tuning."""

    def __init__(
        self,
        data_path: Path,
        model_output_path: Path,
        evaluation_output_path: Optional[Path] = None,
        target_column: str = "Churn",
        random_state: int = 42,
        dimension_reduction_method: str = "select_from_model",
        dimension_reduction_k: Optional[int] = None,
        use_scaling: bool = True,
        use_smote: bool = True,
        create_ensemble: bool = True,
    ):
        """
        Initialize advanced ML pipeline.

        Args:
            data_path: Path to raw data file
            model_output_path: Path to save trained model
            evaluation_output_path: Path to save evaluation report
            target_column: Name of target column
            random_state: Random state for reproducibility
            dimension_reduction_method: Method for dimension reduction
            dimension_reduction_k: Number of features/components to keep
            use_scaling: Whether to scale features
            use_smote: Whether to use SMOTE for class balancing
            create_ensemble: Whether to create ensemble model
        """
        self.data_path = Path(data_path)
        self.model_output_path = Path(model_output_path)
        self.evaluation_output_path = (
            evaluation_output_path
            or self.model_output_path.parent / "evaluation_report.json"
        )
        self.model_comparison_path = (
            self.model_output_path.parent / "model_comparison.json"
        )
        self.reducer_output_path = (
            self.model_output_path.parent / "dimension_reducer.joblib"
        )
        self.visualizations_dir = self.model_output_path.parent / "visualizations"
        self.target_column = target_column
        self.random_state = random_state
        self.use_scaling = use_scaling
        self.use_smote = use_smote
        self.create_ensemble = create_ensemble

        # Initialize visualizer
        self.visualizer = ModelVisualizer(self.visualizations_dir)

        # Initialize dimension reducer
        self.dimension_reducer = DimensionReducer(
            method=dimension_reduction_method,
            k_best=dimension_reduction_k,
            random_state=random_state,
        )

        # Initialize components
        self.data_ingestion = DataIngestion(self.data_path, target_column)
        self.data_preprocessor = DataPreprocessor(target_column, self.dimension_reducer)
        self.model_trainer = AdvancedModelTrainer(
            random_state=random_state, use_scaling=use_scaling, use_smote=use_smote
        )
        self.model_evaluator = ModelEvaluator()

        # Pipeline state
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.trained_model: Optional[Any] = None
        self.pipeline_results: Dict[str, Any] = {}

    def run(self) -> bool:
        """
        Run the complete advanced ML pipeline.

        Returns:
            True if pipeline completed successfully, False otherwise
        """
        try:
            logger.info("=" * 70)
            logger.info("STARTING ADVANCED ML PIPELINE")
            logger.info("=" * 70)

            # Step 1: Data Ingestion
            logger.info("\n[1/6] Data Ingestion")
            self.raw_data = self.data_ingestion.load_data()
            if self.raw_data is None:
                logger.error("Failed to load data")
                return False

            if not self.data_ingestion.validate_data():
                logger.error("Data validation failed")
                return False

            # Step 2: Data Preprocessing
            logger.info("\n[2/6] Data Preprocessing")
            self.processed_data = self.data_preprocessor.preprocess(self.raw_data)
            if self.processed_data is None:
                logger.error("Data preprocessing failed")
                return False

            # Step 3: Prepare Training Data
            logger.info("\n[3/6] Preparing Training Data")
            if not self.model_trainer.prepare_data(
                self.processed_data, self.target_column
            ):
                logger.error("Failed to prepare training data")
                return False

            # Step 4: Train All Models with GridSearchCV
            logger.info("\n[4/6] Training All Models with GridSearchCV")
            model_results = self.model_trainer.train_all_models()
            if not model_results:
                logger.error("Model training failed")
                return False

            # Step 5: Create Ensemble (optional)
            if self.create_ensemble:
                logger.info("\n[5/6] Creating Ensemble Model")
                ensemble = self.model_trainer.train_ensemble()
                if ensemble:
                    logger.info("Ensemble model created successfully")

            # Step 6: Evaluate Best Model
            logger.info("\n[6/6] Evaluating Best Model")
            best_model = self.model_trainer.get_best_model()
            best_model_name = self.model_trainer.get_best_model_name()

            if best_model is None:
                logger.error("No model available for evaluation")
                return False

            # Get updated model results (may include ensemble)
            model_results = self.model_trainer.get_model_results()

            evaluation_results = self.model_evaluator.evaluate(
                best_model, self.model_trainer.X_test, self.model_trainer.y_test
            )

            # Get feature importance
            feature_names = self.data_preprocessor.get_feature_columns()
            feature_importance = self.model_evaluator.get_feature_importance(
                best_model, feature_names
            )

            # Store trained model
            self.trained_model = best_model

            # Compile pipeline results
            self.pipeline_results = {
                "data_info": self.data_ingestion.get_data_info(),
                "preprocessing_info": self.data_preprocessor.get_preprocessing_info(),
                "training_info": self.model_trainer.get_training_info(),
                "model_comparison": {
                    name: {
                        "test_score": results["test_score"],
                        "cv_mean_score": results["cv_mean_score"],
                        "cv_std_score": results["cv_std_score"],
                        "best_params": results["best_params"],
                    }
                    for name, results in model_results.items()
                },
                "best_model": best_model_name,
                "evaluation": evaluation_results,
                "feature_importance": feature_importance,
            }

            # Save model
            logger.info(
                f"\nSaving best model ({best_model_name}) to {self.model_output_path}"
            )
            self.save_model()

            # Save dimension reducer
            if (
                self.dimension_reducer is not None
                and self.dimension_reducer.reducer is not None
            ):
                logger.info(f"Saving dimension reducer to {self.reducer_output_path}")
                self.dimension_reducer.save(self.reducer_output_path)

            # Save evaluation report
            logger.info(f"Saving evaluation report to {self.evaluation_output_path}")
            self.model_evaluator.save_evaluation_report(self.evaluation_output_path)

            # Save model comparison
            logger.info(f"Saving model comparison to {self.model_comparison_path}")
            self.save_model_comparison()

            # Create visualizations
            logger.info("\nCreating visualizations...")
            self.visualizer.create_all_visualizations(
                model_results=model_results,
                best_model_name=best_model_name,
                best_model=best_model,
                X_train=self.model_trainer.X_train,
                y_train=self.model_trainer.y_train,
                X_test=self.model_trainer.X_test,
                y_test=self.model_trainer.y_test,
                evaluation_results=evaluation_results,
            )

            logger.info("\n" + "=" * 70)
            logger.info("ADVANCED ML PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

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

            # Get best model results for metadata
            best_model_name = self.model_trainer.get_best_model_name()
            model_results = self.model_trainer.get_model_results()
            best_results = model_results.get(best_model_name, {})

            # Build a dynamic filename based on the best model name
            safe_name = (best_model_name or "model").replace(" ", "_")
            model_dir = self.model_output_path.parent
            model_dir.mkdir(parents=True, exist_ok=True)
            dynamic_model_path = model_dir / f"{safe_name}.joblib"

            # Save model with comprehensive metadata
            save_dict = {
                "model": self.trained_model,
                "model_name": best_model_name,
                "scaler": self.model_trainer.scaler,
                "feature_names": self.data_preprocessor.get_feature_columns(),
                "test_accuracy": best_results.get("test_score", 0),
                "cv_mean_accuracy": best_results.get("cv_mean_score", 0),
                "cv_std_accuracy": best_results.get("cv_std_score", 0),
                "best_parameters": best_results.get("best_params", {}),
                "training_info": self.model_trainer.get_training_info(),
            }

            joblib.dump(save_dict, dynamic_model_path)
            logger.info(f"Best model ({best_model_name}) saved to {dynamic_model_path}")
            logger.info(f"  Test Accuracy: {best_results.get('test_score', 0):.4f}")
            logger.info(
                f"  CV Mean Accuracy: {best_results.get('cv_mean_score', 0):.4f}"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            return False

    def save_model_comparison(self) -> bool:
        """
        Save model comparison results.

        Returns:
            True if successful, False otherwise
        """
        try:
            comparison_data = {
                "best_model": self.model_trainer.get_best_model_name(),
                "models": {},
            }

            for name, results in self.model_trainer.get_model_results().items():
                comparison_data["models"][name] = {
                    "test_accuracy": results["test_score"],
                    "cv_mean_accuracy": results["cv_mean_score"],
                    "cv_std_accuracy": results["cv_std_score"],
                    "best_parameters": results["best_params"],
                }

            # Sort by test accuracy
            sorted_models = sorted(
                comparison_data["models"].items(),
                key=lambda x: x[1]["test_accuracy"],
                reverse=True,
            )

            comparison_data["ranking"] = [name for name, _ in sorted_models]

            with open(self.model_comparison_path, "w") as f:
                json.dump(comparison_data, f, indent=2, default=str)

            logger.info(f"Model comparison saved to {self.model_comparison_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model comparison: {str(e)}", exc_info=True)
            return False

    def get_results(self) -> Dict[str, Any]:
        """Get pipeline results."""
        return self.pipeline_results.copy()
