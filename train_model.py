"""Main script to train ML models with GridSearchCV and multiple classifiers."""
import sys
from pathlib import Path
import logging

from ml_pipeline.advanced_pipeline import AdvancedMLPipeline

# Import logging config function
try:
    from ml_pipeline.core.logging_config import setup_logging
except ImportError:
    # Fallback if import fails
    logging.basicConfig(level=logging.INFO)
    def setup_logging():
        pass

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Main function to run the advanced ML training pipeline."""
    try:
        # Get project root
        project_root = Path(__file__).parent
        
        # Define paths
        data_path = project_root / "data" / "raw" / "data.csv"
        model_output_path = project_root / "models" / "best_model.joblib"
        evaluation_output_path = project_root / "models" / "evaluation_report.json"
        
        # Validate data path exists
        if not data_path.exists():
            logger.error(f"Data file not found at {data_path}")
            logger.error("Please ensure the data file exists at data/raw/data.csv")
            sys.exit(1)
        
        logger.info("=" * 70)
        logger.info("ADVANCED ML TRAINING PIPELINE")
        logger.info("=" * 70)
        logger.info("This will train multiple models with GridSearchCV")
        logger.info("Expected training time: 10-30 minutes depending on hardware")
        logger.info("=" * 70)
        
        # Initialize and run advanced pipeline
        pipeline = AdvancedMLPipeline(
            data_path=data_path,
            model_output_path=model_output_path,
            evaluation_output_path=evaluation_output_path,
            target_column="Churn",
            random_state=42,
            dimension_reduction_method="select_from_model",  # Options: "pca", "selectkbest", "mutual_info", "select_from_model", "rfe", "none"
            dimension_reduction_k=None,  # None = auto-select, or specify number
            use_scaling=True,  # Scale features for better performance
            use_smote=True,  # Balance classes using SMOTE
            create_ensemble=True  # Create ensemble of top models
        )
        
        # Run pipeline
        success = pipeline.run()
        
        if success:
            logger.info("\n" + "=" * 70)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            results = pipeline.get_results()
            
            logger.info(f"\nBest Model: {results['best_model']}")
            logger.info(f"Test Accuracy: {results['evaluation']['accuracy']:.4f}")
            logger.info(f"Precision: {results['evaluation']['precision']:.4f}")
            logger.info(f"Recall: {results['evaluation']['recall']:.4f}")
            logger.info(f"F1 Score: {results['evaluation']['f1_score']:.4f}")
            logger.info(f"ROC AUC: {results['evaluation']['roc_auc']:.4f}")
            
            logger.info("\nModel Comparison:")
            for model_name, metrics in results['model_comparison'].items():
                logger.info(f"  {model_name}: {metrics['test_score']:.4f}")
            
            logger.info(f"\nModel comparison saved to: models/model_comparison.json")
            logger.info(f"Evaluation report saved to: models/evaluation_report.json")
            
            sys.exit(0)
        else:
            logger.error("\nTraining failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

