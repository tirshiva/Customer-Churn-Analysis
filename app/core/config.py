"""Application configuration."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Customer Churn Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODEL_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "Scripts"
    TEMPLATES_DIR: Path = BASE_DIR / "app" / "templates"
    STATIC_DIR: Path = BASE_DIR / "app" / "static"
    
    # Model
    MODEL_PATH: Path = MODEL_DIR / "best_model.joblib"  # Updated to use best_model.joblib
    MODEL_NAME: str = "best_model.joblib"
    REDUCER_PATH: Path = MODEL_DIR / "dimension_reducer.joblib"
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True


settings = Settings()

