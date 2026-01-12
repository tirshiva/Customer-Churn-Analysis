# Setup Guide

This guide will help you set up the Customer Churn Analysis project from scratch.

## Prerequisites

- Python 3.11 or higher
- pip or uv (recommended for faster installs)
- Git

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Customer-Churn-Analysis
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n churn-analysis python=3.11
conda activate churn-analysis
```

### 3. Install Dependencies

**Basic installation:**
```bash
pip install -r requirements.txt
```

**Development installation (includes dev tools):**
```bash
pip install -e ".[dev]"
# Or using Makefile:
make install-dev
```

**Using uv (faster):**
```bash
uv pip install -r requirements.txt
```

### 4. Setup Pre-commit Hooks (Optional but Recommended)

```bash
make setup-pre-commit
# Or manually:
pre-commit install
```

### 5. Prepare Data

Place your data file at:
```
data/raw/data.csv
```

Ensure the CSV file has a "Churn" column as the target variable.

### 6. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.11+

# Run tests
make test

# Check code formatting
make lint
```

## Quick Start

### Train Model

```bash
make train
# Or
python train_model.py
```

### Run API

```bash
make run
# Or
python run.py
```

Then visit:
- API: http://localhost:8000/docs
- Web UI: http://localhost:8000

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure virtual environment is activated
2. **Missing data**: Ensure `data/raw/data.csv` exists
3. **Port already in use**: Change port in `run.py` or kill existing process
4. **Memory errors during training**: Reduce parameter grid sizes in `model_trainer.py`

### Getting Help

- Check logs in `logs/` directory
- Review documentation in `docs/` directory
- Check GitHub issues

## Next Steps

- Read [README.md](../README.md) for project overview
- Review [CI_CD.md](CI_CD.md) for deployment
- Check [DOCKER.md](DOCKER.md) for containerization
