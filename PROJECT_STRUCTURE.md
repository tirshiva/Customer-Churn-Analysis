# Project Structure

This document describes the organization and purpose of each directory and file in the project.

## Directory Structure

```
Customer-Churn-Analysis/
├── app/                      # FastAPI web application
│   ├── api/                  # API route handlers
│   │   └── routes.py        # API endpoints
│   ├── core/                 # Core application components
│   │   ├── config.py        # Application settings
│   │   └── logging_config.py # Logging setup (uses shared config)
│   ├── services/             # Business logic layer
│   │   ├── data_processor.py    # Data processing service
│   │   └── prediction_service.py # Model prediction service
│   ├── templates/            # HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── result.html
│   │   └── error.html
│   ├── static/               # Static assets (CSS, JS, images)
│   ├── main.py               # FastAPI application entry point
│   └── __init__.py
│
├── ml_pipeline/              # Machine learning pipeline
│   ├── core/                 # Shared utilities
│   │   └── logging_config.py # Unified logging configuration
│   ├── data_ingestion.py     # Data loading and validation
│   ├── data_preprocessing.py # Data cleaning and feature engineering
│   ├── dimension_reduction.py # Feature selection techniques
│   ├── model_trainer.py      # Model training with GridSearchCV
│   ├── model_evaluator.py     # Model evaluation and metrics
│   ├── model_visualizer.py    # Performance visualizations
│   ├── pipeline.py           # Basic ML pipeline
│   ├── advanced_pipeline.py  # Advanced pipeline with ensemble
│   └── __init__.py
│
├── data/                     # Data directory
│   └── raw/                  # Raw data files
│       └── data.csv          # Input dataset
│
├── models/                    # Trained models and artifacts
│   ├── visualizations/       # Generated plots
│   │   ├── model_comparison.png
│   │   ├── best_model_performance.png
│   │   ├── confusion_matrix.png
│   │   └── metrics_comparison.png
│   ├── *_model.joblib        # Trained models (dynamic naming)
│   ├── dimension_reducer.joblib
│   ├── evaluation_report.json
│   └── model_comparison.json
│
├── tests/                    # Test suite
│   ├── test_data_ingestion.py
│   └── __init__.py
│
├── docs/                     # Documentation
│   ├── SETUP.md             # Setup instructions
│   ├── CI_CD.md              # CI/CD guide
│   ├── DOCKER.md             # Docker deployment
│   ├── DIMENSION_REDUCTION.md # Feature selection guide
│   └── VISUALIZATIONS.md      # Visualization guide
│
├── logs/                     # Application logs
│   ├── app.log               # Application logs
│   └── ml_pipeline.log       # ML pipeline logs
│
├── .gitignore                # Git ignore rules
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # MIT License
├── Makefile                  # Common commands
├── pyproject.toml            # Project configuration (PEP 518)
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker Compose configuration
├── README.md                 # Project documentation
├── PROJECT_STRUCTURE.md      # This file
├── train_model.py            # Model training script
└── run.py                    # API server entry point
```

## Key Files

### Configuration Files

- **pyproject.toml**: Modern Python project configuration (PEP 518/621)
  - Project metadata
  - Dependencies
  - Tool configurations (Black, isort, mypy, pytest, coverage)

- **requirements.txt**: Python package dependencies
  - Production dependencies
  - Development dependencies in pyproject.toml

- **pytest.ini**: Pytest configuration
  - Test paths and patterns
  - Coverage settings
  - Markers for test categorization

- **.pre-commit-config.yaml**: Pre-commit hooks
  - Code formatting (Black, isort)
  - Linting (flake8)
  - Type checking (mypy)
  - File checks

### Application Files

- **app/main.py**: FastAPI application
  - App initialization
  - Middleware setup
  - Route registration

- **app/core/config.py**: Application settings
  - Environment variables
  - Path configurations
  - Model settings

- **train_model.py**: Model training entry point
  - Pipeline initialization
  - Training execution
  - Results reporting

- **run.py**: API server entry point
  - Uvicorn server configuration
  - Development mode settings

### ML Pipeline Files

- **ml_pipeline/advanced_pipeline.py**: Main training pipeline
  - Orchestrates all ML steps
  - Model selection
  - Ensemble creation

- **ml_pipeline/model_trainer.py**: Model training
  - Multiple algorithms
  - Hyperparameter tuning
  - Cross-validation

- **ml_pipeline/model_evaluator.py**: Model evaluation
  - Metrics calculation
  - Feature importance
  - Report generation

## Best Practices Implemented

✅ **Modular Architecture**: Clear separation of concerns  
✅ **Configuration Management**: Centralized settings  
✅ **Logging**: Unified logging across components  
✅ **Testing**: Test suite with coverage  
✅ **Code Quality**: Pre-commit hooks, linting, formatting  
✅ **Documentation**: Comprehensive docs and README  
✅ **Docker Support**: Containerized deployment  
✅ **CI/CD Ready**: Configuration for automation  
✅ **Type Hints**: Python type annotations  
✅ **Error Handling**: Robust error management  

## Data Flow

```
data/raw/data.csv
    ↓
ml_pipeline/data_ingestion.py
    ↓
ml_pipeline/data_preprocessing.py
    ↓
ml_pipeline/dimension_reduction.py
    ↓
ml_pipeline/model_trainer.py
    ↓
ml_pipeline/model_evaluator.py
    ↓
models/*_model.joblib
    ↓
app/services/prediction_service.py
    ↓
app/api/routes.py
    ↓
User (via API/Web UI)
```

## Adding New Features

1. **New ML Model**: Add to `ml_pipeline/model_trainer.py`
2. **New API Endpoint**: Add to `app/api/routes.py`
3. **New Service**: Add to `app/services/`
4. **New Test**: Add to `tests/`
5. **Documentation**: Update relevant `docs/` files

## Notes

- All model files use dynamic naming based on best model
- Logs are separated by component (app.log, ml_pipeline.log)
- Data directory structure follows ML best practices
- Documentation is organized in `docs/` directory
- Configuration follows modern Python standards (pyproject.toml)
