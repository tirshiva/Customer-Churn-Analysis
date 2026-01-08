"""Advanced model training module with multiple classifiers and GridSearchCV."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    VotingClassifier, BaggingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Optional, Dict, Any, List, Tuple
import logging
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedModelTrainer:
    """Advanced model trainer with multiple classifiers and hyperparameter tuning."""
    
    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_scaling: bool = True,
        use_smote: bool = True,
        n_jobs: int = -1
    ):
        """
        Initialize advanced model trainer.
        
        Args:
            random_state: Random state for reproducibility
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            use_scaling: Whether to scale features
            use_smote: Whether to use SMOTE for class balancing
            n_jobs: Number of parallel jobs for GridSearchCV
        """
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.use_scaling = use_scaling
        self.use_smote = use_smote
        self.n_jobs = n_jobs
        
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.scaler: Optional[StandardScaler] = None
        
        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.model_results: Dict[str, Dict[str, Any]] = {}
        self.training_info: Dict[str, Any] = {}
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = "Churn"
    ) -> bool:
        """
        Prepare data for training with scaling and balancing.
        
        Args:
            data: Preprocessed DataFrame
            target_column: Name of target column
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if data is None or data.empty:
                logger.error("Data is None or empty")
                return False
            
            if target_column not in data.columns:
                logger.error(f"Target column '{target_column}' not found")
                return False
            
            # Separate features and target
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            logger.info(f"Data split completed:")
            logger.info(f"  Training set: {self.X_train.shape}")
            logger.info(f"  Test set: {self.X_test.shape}")
            logger.info(f"  Training target distribution: {self.y_train.value_counts().to_dict()}")
            logger.info(f"  Test target distribution: {self.y_test.value_counts().to_dict()}")
            
            # Scale features if enabled
            if self.use_scaling:
                logger.info("Scaling features...")
                self.scaler = StandardScaler()
                self.X_train = pd.DataFrame(
                    self.scaler.fit_transform(self.X_train),
                    columns=self.X_train.columns,
                    index=self.X_train.index
                )
                self.X_test = pd.DataFrame(
                    self.scaler.transform(self.X_test),
                    columns=self.X_test.columns,
                    index=self.X_test.index
                )
                logger.info("Features scaled successfully")
            
            # Apply SMOTE for class balancing if enabled
            if self.use_smote:
                try:
                    logger.info("Applying SMOTE for class balancing...")
                    # Check if we have enough samples for SMOTE
                    min_class_count = pd.Series(self.y_train).value_counts().min()
                    if min_class_count < 3:
                        logger.warning(f"Not enough samples for SMOTE (min class: {min_class_count}). Skipping SMOTE.")
                    else:
                        smote = SMOTE(random_state=self.random_state, k_neighbors=min(3, min_class_count - 1))
                        X_train_resampled, y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
                        # Convert back to DataFrame
                        self.X_train = pd.DataFrame(
                            X_train_resampled,
                            columns=self.X_train.columns,
                            index=range(len(X_train_resampled))
                        )
                        self.y_train = pd.Series(y_train_resampled, index=self.X_train.index)
                        logger.info(f"After SMOTE - Training set: {self.X_train.shape}")
                        logger.info(f"After SMOTE - Target distribution: {self.y_train.value_counts().to_dict()}")
                except Exception as e:
                    logger.warning(f"SMOTE failed: {str(e)}. Continuing without SMOTE.")
                    self.use_smote = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}", exc_info=True)
            return False
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configurations for all models with hyperparameter grids.
        
        Returns:
            Dictionary of model configurations
        """
        configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 1.0]
                }
            },
            'adaboost': {
                'model': AdaBoostClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=-1),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'lbfgs', 'saga']
                }
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
            },
            'knn': {
                'model': KNeighborsClassifier(n_jobs=-1),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=self.random_state, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Filter out incompatible parameter combinations
        filtered_configs = {}
        for name, config in configs.items():
            model = config['model']
            params = config['params'].copy()
            
            # Remove incompatible combinations
            if name == 'logistic_regression':
                # Remove elasticnet if solver doesn't support it
                if 'elasticnet' in params.get('penalty', []):
                    valid_solvers = ['saga']
                    params['solver'] = [s for s in params.get('solver', []) if s in valid_solvers]
            
            filtered_configs[name] = {
                'model': model,
                'params': params
            }
        
        return filtered_configs
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train all models with GridSearchCV and return results.
        
        Returns:
            Dictionary with results for each model
        """
        try:
            if self.X_train is None or self.y_train is None:
                logger.error("Data not prepared. Call prepare_data() first.")
                return {}
            
            model_configs = self.get_model_configs()
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            logger.info("=" * 70)
            logger.info("TRAINING ALL MODELS WITH GRIDSEARCHCV")
            logger.info("=" * 70)
            
            for model_name, config in model_configs.items():
                logger.info(f"\n{'='*70}")
                logger.info(f"Training {model_name.upper()}...")
                logger.info(f"{'='*70}")
                
                try:
                    # Create GridSearchCV
                    grid_search = GridSearchCV(
                        estimator=config['model'],
                        param_grid=config['params'],
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=self.n_jobs,
                        verbose=1
                    )
                    
                    # Train model
                    grid_search.fit(self.X_train, self.y_train)
                    
                    # Get best model
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    
                    # Evaluate on test set
                    test_score = best_model.score(self.X_test, self.y_test)
                    
                    # Cross-validation scores
                    cv_scores = cross_val_score(
                        best_model, self.X_train, self.y_train,
                        cv=cv, scoring='accuracy'
                    )
                    
                    # Store results
                    self.models[model_name] = best_model
                    self.model_results[model_name] = {
                        'best_params': best_params,
                        'cv_mean_score': best_score,
                        'cv_std_score': cv_scores.std(),
                        'test_score': test_score,
                        'cv_scores': cv_scores.tolist(),
                        'model': best_model
                    }
                    
                    logger.info(f"\n{model_name.upper()} Results:")
                    logger.info(f"  Best Parameters: {best_params}")
                    logger.info(f"  CV Mean Score: {best_score:.4f} (+/- {cv_scores.std():.4f})")
                    logger.info(f"  Test Score: {test_score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}", exc_info=True)
                    continue
            
            # Find best model
            self._select_best_model()
            
            logger.info("\n" + "=" * 70)
            logger.info("TRAINING COMPLETED")
            logger.info("=" * 70)
            
            return self.model_results
            
        except Exception as e:
            logger.error(f"Error in train_all_models: {str(e)}", exc_info=True)
            return {}
    
    def _select_best_model(self) -> None:
        """Select the best model based on test score."""
        if not self.model_results:
            return
        
        best_score = -1
        best_name = None
        
        for model_name, results in self.model_results.items():
            test_score = results['test_score']
            if test_score > best_score:
                best_score = test_score
                best_name = model_name
        
        if best_name:
            self.best_model = self.models[best_name]
            self.best_model_name = best_name
            logger.info(f"\n{'='*70}")
            logger.info(f"BEST MODEL: {best_name.upper()}")
            logger.info(f"Test Accuracy: {best_score:.4f}")
            logger.info(f"{'='*70}")
    
    def train_ensemble(self) -> Optional[Any]:
        """
        Create and train an ensemble of top models.
        
        Returns:
            Trained ensemble model or None
        """
        try:
            if not self.model_results:
                logger.error("No models trained. Call train_all_models() first.")
                return None
            
            # Select top 3 models
            sorted_models = sorted(
                self.model_results.items(),
                key=lambda x: x[1]['test_score'],
                reverse=True
            )[:3]
            
            if len(sorted_models) < 2:
                logger.warning("Not enough models for ensemble")
                return None
            
            logger.info("\n" + "=" * 70)
            logger.info("CREATING ENSEMBLE MODEL")
            logger.info("=" * 70)
            
            estimators = []
            for model_name, results in sorted_models:
                estimators.append((model_name, results['model']))
                logger.info(f"  Adding {model_name} (Score: {results['test_score']:.4f})")
            
            # Create voting classifier
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
            
            # Train ensemble
            ensemble.fit(self.X_train, self.y_train)
            
            # Evaluate
            ensemble_score = ensemble.score(self.X_test, self.y_test)
            logger.info(f"\nEnsemble Test Score: {ensemble_score:.4f}")
            
            # Compare with best single model
            if ensemble_score > self.model_results[self.best_model_name]['test_score']:
                logger.info("Ensemble performs better than best single model!")
                self.best_model = ensemble
                self.best_model_name = 'ensemble'
            else:
                logger.info("Best single model performs better than ensemble")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}", exc_info=True)
            return None
    
    def get_best_model(self) -> Optional[Any]:
        """Get the best performing model."""
        return self.best_model
    
    def get_best_model_name(self) -> Optional[str]:
        """Get the name of the best model."""
        return self.best_model_name
    
    def get_model_results(self) -> Dict[str, Dict[str, Any]]:
        """Get results for all models."""
        return self.model_results.copy()
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get training information."""
        return {
            'best_model': self.best_model_name,
            'best_test_score': self.model_results.get(self.best_model_name, {}).get('test_score', 0),
            'models_trained': list(self.model_results.keys()),
            'use_scaling': self.use_scaling,
            'use_smote': self.use_smote,
            'cv_folds': self.cv_folds
        }
