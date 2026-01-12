# Customer Churn Analysis - End-to-End ML Project

A comprehensive machine learning project for predicting customer churn using advanced ML techniques, featuring a production-ready FastAPI web application.

## 🎯 Project Overview

This project demonstrates a complete end-to-end machine learning pipeline for customer churn prediction, including:

- **Data Ingestion & Preprocessing**: Automated data cleaning, feature engineering, and handling missing values
- **Advanced ML Pipeline**: Multiple model training with hyperparameter tuning using GridSearchCV
- **Model Selection**: Automatic selection of best performing model with ensemble support
- **Dimension Reduction**: Multiple feature selection techniques (PCA, SelectKBest, Mutual Info, etc.)
- **Class Balancing**: SMOTE for handling imbalanced datasets
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Visualization**: Automated generation of performance visualizations
- **Production API**: FastAPI-based REST API with HTML interface for predictions
- **Docker Support**: Containerized deployment ready

## 📁 Project Structure

```
Customer-Churn-Analysis/
├── app/                    # FastAPI web application
│   ├── api/               # API routes
│   ├── core/              # Configuration and logging
│   ├── services/          # Business logic
│   └── templates/         # HTML templates
├── ml_pipeline/           # ML training pipeline
│   ├── core/             # Shared utilities
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── dimension_reduction.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── model_visualizer.py
│   └── advanced_pipeline.py
├── data/                  # Data directory
│   └── raw/              # Raw data files
├── models/                # Trained models and artifacts
│   └── visualizations/   # Generated plots
├── tests/                 # Unit and integration tests
├── logs/                  # Application logs
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── Makefile              # Common commands

```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or uv
- (Optional) Docker and Docker Compose

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Customer-Churn-Analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or using uv (faster):
   uv pip install -r requirements.txt
   ```

4. **Prepare data**
   - Place your data file at `data/raw/data.csv`
   - Ensure the target column is named "Churn"

### Training the Model

```bash
# Using Makefile
make train

# Or directly
python train_model.py
```

The training pipeline will:
- Load and preprocess data
- Train multiple models (Random Forest, Gradient Boosting, SVM, etc.)
- Perform hyperparameter tuning with GridSearchCV
- Select the best model
- Create an ensemble (optional)
- Generate evaluation reports and visualizations
- Save the best model with dynamic naming

### Running the API

```bash
# Using Makefile
make run

# Or directly
python run.py
# Or
uvicorn app.main:app --reload
```

Access the API at:
- **API Documentation**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8000
- **ReDoc**: http://localhost:8000/redoc

### Using Docker

```bash
# Build and run with Docker Compose
make docker-compose-up

# Or build and run manually
make docker-build
make docker-run
```

## 📊 Model Training Details

### Supported Models

- Random Forest
- Gradient Boosting
- AdaBoost
- Extra Trees
- Logistic Regression
- SVM
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Neural Network (MLP)

### Feature Engineering

- One-hot encoding for categorical variables
- Missing value imputation (median for numerical, mode for categorical)
- Feature scaling (StandardScaler)
- Dimension reduction (SelectFromModel, PCA, SelectKBest, etc.)

### Model Selection

The pipeline automatically:
1. Trains all models with optimized hyperparameters
2. Evaluates each model using cross-validation
3. Selects the best performing model
4. Optionally creates an ensemble of top 3 models
5. Saves the best model with dynamic naming (e.g., `random_forest_model.joblib`)

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ -v --cov

# Run specific test
pytest tests/test_data_ingestion.py -v
```

## 🔧 Development

### Code Quality

```bash
# Format code
make format

# Lint code
make lint
```

### Project Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make train         # Train the model
make test          # Run tests
make lint          # Run linter
make format        # Format code
make clean         # Clean temporary files
```

## 📈 Model Performance

After training, check:
- `models/evaluation_report.json` - Detailed metrics
- `models/model_comparison.json` - Model comparison
- `models/visualizations/` - Performance plots

## 🐳 Docker Deployment

The project includes:
- Multi-stage Dockerfile for optimized builds
- Docker Compose for easy deployment
- Production-ready configuration

## 📚 Documentation

- [CI/CD Guide](docs/CI_CD.md) - Continuous Integration/Deployment
- [Docker Guide](docs/DOCKER.md) - Docker setup and deployment
- [Dimension Reduction](docs/DIMENSION_REDUCTION.md) - Feature selection techniques
- [Visualizations](docs/VISUALIZATIONS.md) - Understanding model outputs

## 🏗️ Architecture

### ML Pipeline Flow

```
Data Ingestion → Preprocessing → Feature Engineering → 
Dimension Reduction → Model Training → Evaluation → 
Model Selection → Visualization → Model Persistence
```

### API Architecture

```
FastAPI Application
├── Routes (API endpoints)
├── Services (Business logic)
│   ├── Prediction Service
│   └── Data Processor
└── Core (Configuration, Logging)
```

## 🔐 Configuration

Configuration is managed through:
- Environment variables (`.env` file)
- `app/core/config.py` - Application settings
- `pyproject.toml` - Project metadata

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- scikit-learn for ML algorithms
- FastAPI for the web framework
- All open-source contributors

## 📊 Project Highlights

✅ **End-to-End Pipeline**: Complete ML workflow from data to deployment  
✅ **Production Ready**: FastAPI API with Docker support  
✅ **Best Practices**: Clean code, testing, documentation  
✅ **Scalable**: Modular architecture, easy to extend  
✅ **Comprehensive**: Multiple models, hyperparameter tuning, ensemble methods  
✅ **Visualization**: Automated performance plots  
✅ **Documentation**: Well-documented code and guides  

---

**Note**: This project demonstrates professional ML engineering practices suitable for production environments.
