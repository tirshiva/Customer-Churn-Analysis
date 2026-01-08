"""Prediction service."""
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from app.core.config import settings
from app.services.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class PredictionService:
    """Handle model predictions."""
    
    def __init__(self):
        """Initialize the prediction service."""
        self.model = None
        self.feature_names = None
        self.data_processor = DataProcessor()
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            if not settings.MODEL_PATH.exists():
                logger.error(f"Model file not found at {settings.MODEL_PATH}")
                return
            
            loaded_data = joblib.load(settings.MODEL_PATH)
            
            # Handle both old format (just model) and new format (dict with metadata)
            if isinstance(loaded_data, dict):
                self.model = loaded_data.get('model')
                self.feature_names = loaded_data.get('feature_names')
                self.scaler = loaded_data.get('scaler')
                model_name = loaded_data.get('model_name', 'unknown')
                logger.info(f"Loaded model: {model_name}")
            else:
                # Old format - just the model
                self.model = loaded_data
                self.scaler = None
                # Try to get feature names from model
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = self.model.feature_names_in_.tolist()
                else:
                    logger.warning("Model does not have feature_names_in_ attribute")
                    self.feature_names = None
            
            logger.info(f"Model loaded successfully from {settings.MODEL_PATH}")
            logger.info(f"Model type: {type(self.model).__name__}")
            if self.feature_names:
                logger.info(f"Number of features: {len(self.feature_names)}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            self.model = None
            self.feature_names = None
            self.scaler = None
    
    def is_ready(self) -> bool:
        """Check if the service is ready for predictions."""
        return self.model is not None and self.feature_names is not None
    
    def predict(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a prediction for the given input data.
        
        Args:
            input_data: Dictionary containing customer features
            
        Returns:
            Dictionary with prediction results or None if error occurs
        """
        if not self.is_ready():
            logger.error("Model is not loaded. Cannot make predictions.")
            return None
        
        try:
            # Preprocess input data
            processed_data = self.data_processor.preprocess_input(input_data)
            if processed_data is None:
                logger.error("Failed to preprocess input data")
                return None
            
            # Align features with model
            aligned_data = self.data_processor.align_features(
                processed_data, 
                self.feature_names
            )
            if aligned_data is None:
                logger.error("Failed to align features")
                return None
            
            # Make prediction
            prediction_proba = self.model.predict_proba(aligned_data)
            prediction_class = self.model.predict(aligned_data)
            
            # Extract probabilities
            churn_probability = float(prediction_proba[0][1]) * 100
            no_churn_probability = float(prediction_proba[0][0]) * 100
            predicted_class = int(prediction_class[0])
            
            # Determine risk level
            if churn_probability < 30:
                risk_level = "Low"
                risk_color = "success"
            elif churn_probability < 70:
                risk_level = "Medium"
                risk_color = "warning"
            else:
                risk_level = "High"
                risk_color = "danger"
            
            result = {
                "churn_probability": round(churn_probability, 2),
                "no_churn_probability": round(no_churn_probability, 2),
                "predicted_class": predicted_class,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "status": "success"
            }
            
            logger.info(f"Prediction successful. Churn probability: {churn_probability:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_feature_importance(self, top_n: int = 10) -> Optional[list]:
        """
        Get top feature importances from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of tuples (feature_name, importance) or None if error
        """
        if not self.is_ready():
            return None
        
        try:
            if not hasattr(self.model, 'feature_importances_'):
                logger.warning("Model does not support feature importances")
                return None
            
            importances = self.model.feature_importances_
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return feature_importance[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}", exc_info=True)
            return None

