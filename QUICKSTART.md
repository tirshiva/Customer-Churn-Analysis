# Quick Start Guide

Complete step-by-step guide to set up and run the Customer Churn Prediction project.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Training the ML Model](#training-the-ml-model)
5. [Running the Application](#running-the-application)
6. [Using Docker](#using-docker)
7. [How Components Work](#how-components-work)
8. [Testing](#testing)
9. [Usage Guide](#usage-guide)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Python 3.8 or higher**
   ```bash
   python --version  # Should show 3.8+
   ```

2. **pip** (Python package manager)
   ```bash
   pip --version
   ```

3. **Git** (for cloning the repository)
   ```bash
   git --version
   ```

### Optional (Recommended)

- **Docker & Docker Compose** (for containerized deployment)
  ```bash
  docker --version
  docker-compose --version
  ```

- **Virtual Environment** (highly recommended)
  - `venv` (built-in with Python)
  - `conda` (if using Anaconda/Miniconda)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Customer-Churn-Analysis
```

### Step 2: Create Virtual Environment (Recommended)

**Using venv:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n churn-analysis python=3.11
conda activate churn-analysis
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**What gets installed:**
- FastAPI and Uvicorn (web framework and server)
- scikit-learn, pandas, numpy (ML and data processing)
- Pydantic (data validation)
- pytest (testing framework)
- And more (see `requirements.txt`)

### Step 4: Verify Installation

```bash
# Check if packages are installed
python -c "import fastapi; print('✅ FastAPI installed')"
python -c "import sklearn; print('✅ scikit-learn installed')"
python -c "import pandas; print('✅ pandas installed')"
```

### Step 5: Prepare Data File

Ensure your training data is in the correct location:

```bash
# Check if data file exists
ls Scripts/data.csv  # Linux/Mac
dir Scripts\data.csv  # Windows

# If data file is in artifacts/, move it:
# mv artifacts/data.csv Scripts/data.csv
```

**Data file requirements:**
- CSV format
- Must contain a "Churn" column (target variable)
- Should have customer features (tenure, MonthlyCharges, etc.)

---

## Project Structure

Understanding the project structure helps you navigate the codebase:

```
Customer-Churn-Analysis/
├── app/                    # FastAPI Web Application
│   ├── main.py            # Application entry point
│   ├── api/               # API routes
│   │   └── routes.py      # Root (/) and /predict endpoints
│   ├── core/              # Core configuration
│   │   ├── config.py      # App settings and paths
│   │   └── logging_config.py  # Logging setup
│   ├── services/          # Business logic
│   │   ├── data_processor.py      # Data preprocessing for predictions
│   │   └── prediction_service.py  # ML model prediction service
│   └── templates/         # HTML templates
│       ├── index.html     # Prediction form
│       ├── result.html    # Results page
│       └── error.html     # Error page
│
├── ml_pipeline/           # ML Training Pipeline (OOP-based)
│   ├── data_ingestion.py  # Loads and validates data
│   ├── data_preprocessing.py  # Cleans and transforms data
│   ├── model_trainer.py   # Trains ML models
│   ├── model_evaluator.py # Evaluates model performance
│   ├── pipeline.py       # Orchestrates complete pipeline
│   └── core/
│       └── logging_config.py  # ML pipeline logging
│
├── models/                # Trained ML models
│   └── random_forest_model.joblib  # Trained model (created after training)
│
├── Scripts/               # Data files
│   └── data.csv          # Training data
│
├── tests/                 # Unit tests
│   └── test_data_ingestion.py
│
├── logs/                  # Application logs (created automatically)
│
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── train_model.py         # ML training entry point
├── run.py                 # Application runner
└── Makefile               # Common commands
```

---

## Training the ML Model

### Overview

The ML pipeline uses an **Object-Oriented Programming (OOP)** approach with these stages:

1. **Data Ingestion** → Loads raw data from CSV
2. **Data Preprocessing** → Cleans and transforms data
3. **Model Training** → Trains Random Forest classifier
4. **Model Evaluation** → Calculates comprehensive metrics
5. **Model Persistence** → Saves trained model and evaluation report

### Step 1: Ensure Data File Exists

```bash
# Check if data file exists
python -c "from pathlib import Path; print('✅ Data found' if Path('Scripts/data.csv').exists() else '❌ Data not found')"
```

### Step 2: Train the Model

**Option 1: Using Python script (Recommended)**
```bash
python train_model.py
```

**Option 2: Using Make**
```bash
make train
```

### Step 3: What Happens During Training

The training process will:

1. **Load Data** (`DataIngestion`)
   - Reads `Scripts/data.csv`
   - Validates data structure
   - Logs data statistics

2. **Preprocess Data** (`DataPreprocessor`)
   - Handles missing values
   - Encodes categorical variables (one-hot encoding)
   - Converts target variable to numeric

3. **Split Data** (`ModelTrainer`)
   - 80% training, 20% testing
   - Stratified split (maintains class distribution)

4. **Train Model** (`ModelTrainer`)
   - Trains Random Forest classifier
   - Performs cross-validation
   - Logs training progress

5. **Evaluate Model** (`ModelEvaluator`)
   - Calculates accuracy, precision, recall, F1-score, ROC-AUC
   - Generates confusion matrix
   - Computes feature importance

6. **Save Results**
   - Model saved to: `models/random_forest_model.joblib`
   - Evaluation report: `models/evaluation_report.json`

### Step 4: Verify Training Success

```bash
# Check if model was created
ls models/random_forest_model.joblib

# View evaluation report
cat models/evaluation_report.json  # Linux/Mac
type models\evaluation_report.json  # Windows
```

**Expected Output:**
- Model file: `models/random_forest_model.joblib`
- Evaluation report: `models/evaluation_report.json`
- Logs: `logs/ml_pipeline.log`

### Training Output Example

```
============================================================
STARTING ML PIPELINE
============================================================

[1/5] Data Ingestion
Data loaded successfully. Shape: (7043, 21)
Data validation passed

[2/5] Data Preprocessing
Preprocessing completed. Shape: (7043, 46)
Number of features: 45

[3/5] Preparing Training Data
Data split completed:
  Training set: (5634, 45)
  Test set: (1409, 45)

[4/5] Training Model
Training random_forest model...
Cross-validation results: 0.8468 (+/- 0.0123)
Model training completed

[5/5] Evaluating Model
==================================================
MODEL EVALUATION RESULTS
==================================================
Accuracy:  0.8468
Precision: 0.7234
Recall:    0.5123
F1 Score:  0.5987
ROC AUC:   0.8456
==================================================

Model saved to models/random_forest_model.joblib
Evaluation report saved to models/evaluation_report.json

============================================================
ML PIPELINE COMPLETED SUCCESSFULLY
============================================================
```

---

## Running the Application

### Prerequisites

Before running the application, ensure:
- ✅ Model is trained (`models/random_forest_model.joblib` exists)
- ✅ Dependencies are installed
- ✅ Virtual environment is activated (if using)

### Option 1: Using run.py (Recommended)

```bash
python run.py
```

**What it does:**
- Starts Uvicorn server
- Loads FastAPI application
- Serves on `http://localhost:8000`

### Option 2: Using uvicorn directly

**Development mode (with auto-reload):**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option 3: Using Make

```bash
# Start application (uses run.py)
make run  # If added to Makefile, or:
python run.py
```

### Access the Application

Once running, access:

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

### Application Startup Process

1. **Load Configuration** (`app/core/config.py`)
   - Reads settings
   - Sets up paths (model, templates, static files)

2. **Initialize Logging** (`app/core/logging_config.py`)
   - Sets up console and file logging
   - Creates `logs/` directory

3. **Load ML Model** (`app/services/prediction_service.py`)
   - Loads `models/random_forest_model.joblib`
   - Extracts feature names from model
   - Validates model is ready

4. **Start FastAPI Server**
   - Registers routes
   - Starts Uvicorn ASGI server
   - Ready to accept requests

---

## Using Docker

### Prerequisites

- Docker installed and running
- Docker Compose installed

### Quick Start with Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build

```bash
# Build Docker image
docker build -t customer-churn-api:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  customer-churn-api:latest
```

### Docker Compose Configuration

The `docker-compose.yml` includes:
- Port mapping: `8000:8000`
- Volume mounts for models, logs, and data
- Health checks
- Environment variables

**Note:** Ensure model is trained before running Docker, or mount a directory with the model.

---

## How Components Work

### 1. FastAPI Application (`app/`)

**Main Entry Point** (`app/main.py`):
- Creates FastAPI app instance
- Configures CORS middleware
- Includes API routes
- Sets up startup/shutdown events

**API Routes** (`app/api/routes.py`):
- `GET /`: Displays prediction form (HTML)
- `GET /predict`: Displays form (redirect)
- `POST /predict`: Processes form, returns prediction results (HTML)

**Services**:
- `PredictionService`: Loads model, makes predictions
- `DataProcessor`: Preprocesses input data for predictions

**Flow:**
```
User Request → Routes → Services → Model → Response
```

### 2. ML Pipeline (`ml_pipeline/`)

**Data Ingestion** (`data_ingestion.py`):
- Loads CSV file
- Validates data structure
- Provides data statistics

**Data Preprocessing** (`data_preprocessing.py`):
- Handles missing values (median for numeric, mode for categorical)
- One-hot encodes categorical variables
- Converts target to numeric

**Model Training** (`model_trainer.py`):
- Splits data (train/test)
- Trains Random Forest classifier
- Performs cross-validation
- Supports hyperparameter tuning

**Model Evaluation** (`model_evaluator.py`):
- Calculates multiple metrics
- Generates confusion matrix
- Computes feature importance
- Saves evaluation report

**Pipeline Orchestrator** (`pipeline.py`):
- Coordinates all components
- Manages pipeline execution
- Handles errors
- Saves results

### 3. Request Flow

**Prediction Request Flow:**

```
1. User fills form → POST /predict
2. Routes receives form data
3. DataProcessor preprocesses input
4. PredictionService loads model
5. Model makes prediction
6. Results formatted with risk levels
7. HTML response with results
```

**Data Flow:**

```
Raw Data (CSV)
  ↓
Data Ingestion (load & validate)
  ↓
Data Preprocessing (clean & transform)
  ↓
Model Training (train & validate)
  ↓
Model Evaluation (metrics & reports)
  ↓
Trained Model (saved)
```

---

## Testing

### Run All Tests

```bash
# Using pytest
pytest tests/ -v

# Using Make
make test

# With coverage
pytest tests/ -v --cov=app --cov=ml_pipeline --cov-report=html
```

### Test Individual Components

```bash
# Test data ingestion
pytest tests/test_data_ingestion.py -v

# Test with specific markers
pytest -m "not slow" -v
```

### Verify Application Works

```bash
# Test imports
python -c "from app.main import app; print('✅ App OK')"
python -c "from ml_pipeline.pipeline import MLPipeline; print('✅ Pipeline OK')"

# Test model loading
python -c "from app.services.prediction_service import PredictionService; ps = PredictionService(); print('✅ Model loaded' if ps.is_ready() else '❌ Model not ready')"
```

---

## Usage Guide

### Making Predictions

1. **Start the application:**
   ```bash
   python run.py
   ```

2. **Open browser:**
   Navigate to http://localhost:8000

3. **Fill the form:**
   - **Required fields:**
     - Tenure (months)
     - Monthly Charges ($)
     - Total Charges ($)
   - **Optional fields:**
     - Customer demographics
     - Service information
     - Contract details

4. **Submit prediction:**
   Click "Predict Churn Probability"

5. **View results:**
   - Churn probability percentage
   - Risk level (Low/Medium/High)
   - Top factors affecting prediction
   - Customer profile summary

### Understanding Results

**Risk Levels:**
- **Low Risk (< 30%)**: Customer likely to stay
- **Medium Risk (30-70%)**: Moderate churn risk
- **High Risk (> 70%)**: High churn probability

**Feature Importance:**
- Shows top 10 factors affecting the prediction
- Helps understand why customer is at risk

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Not Found Error

**Error:**
```
Model is not loaded. Please ensure the model file exists.
```

**Solution:**
```bash
# Train the model first
python train_model.py

# Verify model exists
ls models/random_forest_model.joblib
```

#### 2. Port Already in Use

**Error:**
```
Address already in use
```

**Solution:**
```bash
# Use a different port
uvicorn app.main:app --port 8080

# Or find and kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
# Linux/Mac:
lsof -ti:8000 | xargs kill
```

#### 3. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'app'
```

**Solution:**
```bash
# Ensure you're in the project root directory
pwd  # Should show Customer-Churn-Analysis

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 4. Data File Not Found

**Error:**
```
Data file not found at Scripts/data.csv
```

**Solution:**
```bash
# Check if data exists in other locations
find . -name "data.csv"

# If found in artifacts/, move it:
mv artifacts/data.csv Scripts/data.csv

# Or update path in train_model.py
```

#### 5. Permission Errors

**Error:**
```
Permission denied
```

**Solution:**
```bash
# Linux/Mac: Fix permissions
chmod +x train_model.py
chmod +x run.py

# Windows: Run as administrator or check file permissions
```

#### 6. Docker Build Fails

**Error:**
```
Docker build fails
```

**Solution:**
```bash
# Check Docker is running
docker ps

# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t customer-churn-api:latest .
```

#### 7. Training Takes Too Long

**Solution:**
```bash
# Reduce data size for testing (modify train_model.py)
# Or use a smaller model:
# Edit ml_pipeline/model_trainer.py to reduce n_estimators
```

### Getting Help

1. **Check logs:**
   ```bash
   # Application logs
   tail -f logs/app.log
   
   # ML pipeline logs
   tail -f logs/ml_pipeline.log
   ```

2. **Verify installation:**
   ```bash
   pip list | grep -E "fastapi|sklearn|pandas"
   ```

3. **Test components individually:**
   ```bash
   # Test data loading
   python -c "from ml_pipeline.data_ingestion import DataIngestion; di = DataIngestion('Scripts/data.csv'); print(di.load_data().shape)"
   ```

---

## Next Steps

After getting the application running:

1. **Explore the API:**
   - Visit http://localhost:8000/docs
   - Try the interactive API documentation
   - Test endpoints directly

2. **Review Evaluation Report:**
   ```bash
   cat models/evaluation_report.json
   ```

3. **Check Logs:**
   ```bash
   tail -f logs/app.log
   ```

4. **Customize:**
   - Modify model parameters in `ml_pipeline/model_trainer.py`
   - Adjust preprocessing in `ml_pipeline/data_preprocessing.py`
   - Update UI in `app/templates/`

5. **Deploy:**
   - Use Docker for production
   - Set up CI/CD (see CI_CD.md)
   - Configure environment variables

---

## Quick Reference

### Essential Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train model
python train_model.py

# Run application
python run.py

# Test
pytest tests/ -v

# Docker
docker-compose up -d
```

### File Locations

- **Model**: `models/random_forest_model.joblib`
- **Data**: `Scripts/data.csv`
- **Logs**: `logs/app.log`, `logs/ml_pipeline.log`
- **Config**: `app/core/config.py`
- **Templates**: `app/templates/`

### Important URLs

- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/

---

## Additional Resources

- **README.md**: Complete project documentation
- **DOCKER.md**: Detailed Docker guide
- **CI_CD.md**: CI/CD pipeline documentation
- **PROJECT_SUMMARY.md**: Project transformation summary
- **CLEANUP_GUIDE.md**: Codebase cleanup guide

---

**Happy Coding! 🚀**

For issues or questions, check the troubleshooting section or review the logs in `logs/` directory.
