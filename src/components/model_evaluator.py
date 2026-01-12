"""Model evaluation module with comprehensive metrics."""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Optional, Dict, Any
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handle comprehensive model evaluation."""

    def __init__(self):
        """Initialize model evaluator."""
        self.evaluation_results: Dict[str, Any] = {}

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # ROC AUC - handle edge cases
            try:
                # Check if we have both classes in y_test
                unique_classes = len(np.unique(y_test))
                if unique_classes < 2:
                    logger.warning("Only one class present in y_test. Cannot calculate ROC AUC.")
                    roc_auc = 0.0
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
            except Exception as e:
                logger.warning(f"Error calculating ROC AUC: {str(e)}. Setting to 0.0")
                roc_auc = 0.0

            # Confusion matrix - handle edge cases
            cm = confusion_matrix(y_test, y_pred)
            try:
                # Check if confusion matrix is 2x2
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                elif cm.shape == (1, 1):
                    # Only one class predicted/actual
                    if y_test.iloc[0] == 0:
                        tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                    else:
                        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                else:
                    # Handle other edge cases
                    logger.warning(f"Unexpected confusion matrix shape: {cm.shape}")
                    tn, fp, fn, tp = 0, 0, 0, 0
            except Exception as e:
                logger.error(f"Error processing confusion matrix: {str(e)}")
                tn, fp, fn, tp = 0, 0, 0, 0

            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Store results
            self.evaluation_results = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                },
                "classification_report": class_report,
                "test_samples": len(y_test),
                "positive_samples": int(y_test.sum()),
                "negative_samples": int(len(y_test) - y_test.sum()),
            }

            # Log results
            logger.info("=" * 50)
            logger.info("MODEL EVALUATION RESULTS")
            logger.info("=" * 50)
            logger.info(f"Accuracy:  {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall:    {recall:.4f}")
            logger.info(f"F1 Score:  {f1:.4f}")
            logger.info(f"ROC AUC:   {roc_auc:.4f}")
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  TN: {tn}, FP: {fp}")
            logger.info(f"  FN: {fn}, TP: {tp}")
            logger.info("=" * 50)

            return self.evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
            return {}

    def get_feature_importance(self, model: Any, feature_names: list) -> Dict[str, float]:
        """
        Get feature importance from model.

        Args:
            model: Trained model
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            # Handle ensemble models (VotingClassifier)
            if hasattr(model, "estimators_"):
                logger.info("Calculating feature importance for ensemble model...")
                all_importances = []

                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, "feature_importances_"):
                        all_importances.append(estimator.feature_importances_)
                    else:
                        logger.debug(f"Estimator {name} does not support feature importances")

                if not all_importances:
                    logger.warning("No estimators in ensemble support feature importances")
                    return {}

                # Average importances across all estimators
                importances = np.mean(all_importances, axis=0)
                feature_importance = dict(zip(feature_names, importances))

                # Sort by importance
                sorted_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )

                logger.info("Top 10 Most Important Features (Ensemble Average):")
                for i, (feature, importance) in enumerate(list(sorted_importance.items())[:10], 1):
                    logger.info(f"  {i}. {feature}: {importance:.4f}")

                return sorted_importance

            # Handle regular models with feature_importances_
            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_importance = dict(zip(feature_names, importances))

                # Sort by importance
                sorted_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )

                logger.info("Top 10 Most Important Features:")
                for i, (feature, importance) in enumerate(list(sorted_importance.items())[:10], 1):
                    logger.info(f"  {i}. {feature}: {importance:.4f}")

                return sorted_importance
            else:
                logger.warning("Model does not support feature importances")
                return {}

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}", exc_info=True)
            return {}

    def save_evaluation_report(self, output_path: Path) -> bool:
        """
        Save evaluation results to JSON file.

        Args:
            output_path: Path to save the report

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(self.evaluation_results, f, indent=2)

            logger.info(f"Evaluation report saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving evaluation report: {str(e)}", exc_info=True)
            return False

    def get_evaluation_results(self) -> Dict[str, Any]:
        """Get evaluation results."""
        return self.evaluation_results.copy()
