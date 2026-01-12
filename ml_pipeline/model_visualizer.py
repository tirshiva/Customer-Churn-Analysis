"""Model visualization module for creating performance graphs."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from sklearn.model_selection import learning_curve, validation_curve

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class ModelVisualizer:
    """Create visualizations for model performance."""

    def __init__(self, output_dir: Path):
        """
        Initialize model visualizer.

        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_model_comparison(
        self, model_results: Dict[str, Dict[str, Any]], save_path: Optional[Path] = None
    ) -> bool:
        """
        Plot comparison of all models.

        Args:
            model_results: Dictionary with model results
            save_path: Path to save the plot

        Returns:
            True if successful, False otherwise
        """
        try:
            if not model_results:
                logger.error("No model results provided")
                return False

            # Extract data
            model_names = []
            test_scores = []
            cv_scores = []

            for name, results in model_results.items():
                model_names.append(name.replace("_", " ").title())
                test_scores.append(results.get("test_score", 0))
                cv_scores.append(results.get("cv_mean_score", 0))

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "Model": model_names,
                    "Test Accuracy": test_scores,
                    "CV Mean Accuracy": cv_scores,
                }
            )

            # Sort by test accuracy
            df = df.sort_values("Test Accuracy", ascending=False)

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 8))
            x = np.arange(len(df))
            width = 0.35

            bars1 = ax.bar(
                x - width / 2,
                df["Test Accuracy"],
                width,
                label="Test Accuracy",
                color="#2ecc71",
                alpha=0.8,
            )
            bars2 = ax.bar(
                x + width / 2,
                df["CV Mean Accuracy"],
                width,
                label="CV Mean Accuracy",
                color="#3498db",
                alpha=0.8,
            )

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            ax.set_xlabel("Models", fontsize=12, fontweight="bold")
            ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
            ax.set_title(
                "Model Comparison: Test Accuracy vs Cross-Validation Accuracy",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(df["Model"], rotation=45, ha="right")
            ax.legend(fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim([0, 1])

            plt.tight_layout()

            save_path = save_path or self.output_dir / "model_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Model comparison plot saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating model comparison plot: {str(e)}", exc_info=True)
            return False

    def plot_best_model_performance(
        self,
        model_name: str,
        test_score: float,
        cv_scores: List[float],
        save_path: Optional[Path] = None,
    ) -> bool:
        """
        Plot performance metrics for the best model.

        Args:
            model_name: Name of the best model
            test_score: Test accuracy score
            cv_scores: List of cross-validation scores
            save_path: Path to save the plot

        Returns:
            True if successful, False otherwise
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Plot 1: Test vs CV Scores
            ax1 = axes[0]
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            categories = ["Test\nAccuracy", "CV Mean\nAccuracy"]
            scores = [test_score, cv_mean]
            errors = [0, cv_std]
            colors = ["#e74c3c", "#3498db"]

            bars = ax1.bar(
                categories,
                scores,
                yerr=errors,
                capsize=10,
                color=colors,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add value labels
            for idx, (bar, score) in enumerate(zip(bars, scores)):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + errors[idx],
                    f"{score:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            ax1.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
            ax1.set_title(
                f'{model_name.replace("_", " ").title()}: Test vs CV Accuracy',
                fontsize=13,
                fontweight="bold",
                pad=15,
            )
            ax1.set_ylim([0, 1])
            ax1.grid(axis="y", alpha=0.3)

            # Plot 2: CV Scores Distribution
            ax2 = axes[1]
            ax2.boxplot(
                cv_scores,
                vert=True,
                patch_artist=True,
                boxprops=dict(facecolor="#3498db", alpha=0.7),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(color="black", linewidth=1.5),
                capprops=dict(color="black", linewidth=1.5),
            )

            ax2.scatter(
                [1] * len(cv_scores),
                cv_scores,
                alpha=0.6,
                color="#2c3e50",
                s=50,
                zorder=3,
            )
            ax2.axhline(
                y=test_score,
                color="#e74c3c",
                linestyle="--",
                linewidth=2,
                label=f"Test Accuracy: {test_score:.4f}",
            )

            ax2.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
            ax2.set_title(
                "Cross-Validation Scores Distribution",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )
            ax2.set_xticklabels(["CV Scores"])
            ax2.set_ylim([0, 1])
            ax2.legend(fontsize=10)
            ax2.grid(axis="y", alpha=0.3)

            plt.suptitle(
                f'Best Model Performance: {model_name.replace("_", " ").title()}',
                fontsize=15,
                fontweight="bold",
                y=1.02,
            )
            plt.tight_layout()

            save_path = save_path or self.output_dir / "best_model_performance.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Best model performance plot saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating best model performance plot: {str(e)}", exc_info=True)
            return False

    def plot_learning_curve(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        save_path: Optional[Path] = None,
        cv_folds: int = 5,
    ) -> bool:
        """
        Plot learning curve for the model.

        Args:
            model: Trained model
            X: Feature DataFrame
            y: Target Series
            model_name: Name of the model
            save_path: Path to save the plot
            cv_folds: Number of CV folds

        Returns:
            True if successful, False otherwise
        """
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model,
                X,
                y,
                cv=cv_folds,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring="accuracy",
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            fig, ax = plt.subplots(figsize=(12, 8))

            ax.plot(
                train_sizes,
                train_mean,
                "o-",
                color="#3498db",
                label="Training Score",
                linewidth=2,
                markersize=8,
            )
            ax.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2,
                color="#3498db",
            )

            ax.plot(
                train_sizes,
                val_mean,
                "o-",
                color="#e74c3c",
                label="Validation Score",
                linewidth=2,
                markersize=8,
            )
            ax.fill_between(
                train_sizes,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.2,
                color="#e74c3c",
            )

            ax.set_xlabel("Training Set Size", fontsize=12, fontweight="bold")
            ax.set_ylabel("Accuracy Score", fontsize=12, fontweight="bold")
            ax.set_title(
                f'Learning Curve: {model_name.replace("_", " ").title()}',
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            ax.legend(loc="best", fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])

            plt.tight_layout()

            save_path = save_path or self.output_dir / "learning_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Learning curve plot saved to {save_path}")
            return True

        except Exception as e:
            logger.warning(f"Error creating learning curve: {str(e)}")
            return False

    def plot_confusion_matrix_heatmap(
        self,
        confusion_matrix: np.ndarray,
        model_name: str,
        save_path: Optional[Path] = None,
    ) -> bool:
        """
        Plot confusion matrix as heatmap.

        Args:
            confusion_matrix: Confusion matrix array
            model_name: Name of the model
            save_path: Path to save the plot

        Returns:
            True if successful, False otherwise
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                cbar_kws={"label": "Count"},
                linewidths=1,
                linecolor="black",
            )

            ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
            ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
            ax.set_title(
                f'Confusion Matrix: {model_name.replace("_", " ").title()}',
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            ax.set_xticklabels(["No Churn", "Churn"])
            ax.set_yticklabels(["No Churn", "Churn"])

            plt.tight_layout()

            save_path = save_path or self.output_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Confusion matrix plot saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {str(e)}", exc_info=True)
            return False

    def plot_metrics_comparison(
        self,
        evaluation_results: Dict[str, float],
        model_name: str,
        save_path: Optional[Path] = None,
    ) -> bool:
        """
        Plot comparison of different metrics.

        Args:
            evaluation_results: Dictionary with metric values
            model_name: Name of the model
            save_path: Path to save the plot

        Returns:
            True if successful, False otherwise
        """
        try:
            metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
            values = [evaluation_results.get(m, 0) for m in metrics]

            fig, ax = plt.subplots(figsize=(12, 7))

            bars = ax.barh(
                metric_labels,
                values,
                color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"],
                alpha=0.8,
            )

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                width = bar.get_width()
                ax.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}",
                    ha="left",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

            ax.set_xlabel("Score", fontsize=12, fontweight="bold")
            ax.set_title(
                f'Performance Metrics: {model_name.replace("_", " ").title()}',
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            ax.set_xlim([0, 1])
            ax.grid(axis="x", alpha=0.3)

            plt.tight_layout()

            save_path = save_path or self.output_dir / "metrics_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Metrics comparison plot saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating metrics comparison plot: {str(e)}", exc_info=True)
            return False

    def create_all_visualizations(
        self,
        model_results: Dict[str, Dict[str, Any]],
        best_model_name: str,
        best_model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        evaluation_results: Dict[str, Any],
    ) -> bool:
        """
        Create all visualizations for the best model.

        Args:
            model_results: Results for all models
            best_model_name: Name of best model
            best_model: Best model instance
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            evaluation_results: Evaluation results dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Creating visualizations for best model...")

            # 1. Model comparison
            self.plot_model_comparison(model_results)

            # 2. Best model performance
            best_results = model_results.get(best_model_name, {})
            cv_scores = best_results.get("cv_scores", [])
            test_score = best_results.get("test_score", 0)
            self.plot_best_model_performance(best_model_name, test_score, cv_scores)

            # 3. Learning curve (skip for ensemble models)
            try:
                # Check if it's an ensemble model
                if hasattr(best_model, "estimators_") and len(best_model.estimators_) > 1:
                    logger.info("Skipping learning curve for ensemble model")
                else:
                    self.plot_learning_curve(best_model, X_train, y_train, best_model_name)
            except Exception as e:
                logger.warning(f"Could not create learning curve: {str(e)}")

            # 4. Confusion matrix
            from sklearn.metrics import confusion_matrix

            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            self.plot_confusion_matrix_heatmap(cm, best_model_name)

            # 5. Metrics comparison
            self.plot_metrics_comparison(evaluation_results, best_model_name)

            logger.info(f"All visualizations saved to {self.output_dir}")
            return True

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}", exc_info=True)
            return False
