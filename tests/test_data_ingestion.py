"""Tests for data ingestion module."""

import pytest
import pandas as pd
from pathlib import Path
from ml_pipeline.data_ingestion import DataIngestion


def test_data_ingestion_initialization():
    """Test DataIngestion initialization."""
    data_path = Path("Scripts/data.csv")
    ingestion = DataIngestion(data_path, target_column="Churn")

    assert ingestion.data_path == data_path
    assert ingestion.target_column == "Churn"
    assert ingestion.raw_data is None


def test_data_ingestion_load_data(tmp_path):
    """Test loading data."""
    # Create a test CSV file
    test_data = pd.DataFrame(
        {"customerID": ["1", "2", "3"], "Churn": ["Yes", "No", "Yes"], "tenure": [10, 20, 30]}
    )
    test_file = tmp_path / "test_data.csv"
    test_data.to_csv(test_file, index=False)

    ingestion = DataIngestion(test_file, target_column="Churn")
    result = ingestion.load_data()

    assert result is not None
    assert len(result) == 3
    assert ingestion.raw_data is not None


def test_data_ingestion_validate_data(tmp_path):
    """Test data validation."""
    test_data = pd.DataFrame({"customerID": ["1", "2"], "Churn": ["Yes", "No"], "tenure": [10, 20]})
    test_file = tmp_path / "test_data.csv"
    test_data.to_csv(test_file, index=False)

    ingestion = DataIngestion(test_file, target_column="Churn")
    ingestion.load_data()
    is_valid = ingestion.validate_data()

    assert is_valid is True
