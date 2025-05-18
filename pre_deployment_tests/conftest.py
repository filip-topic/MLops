import pytest
import pandas as pd
import os

@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    data_path = os.path.join('data', 'Womens Clothing E-Commerce Reviews.csv')
    return pd.read_csv(data_path)

@pytest.fixture
def model_predictions():
    """Mock model predictions for testing."""
    # In a real scenario, this would load actual model predictions
    return [1, 0, 1, 1, 0]

@pytest.fixture
def true_labels():
    """Mock true labels for testing."""
    # In a real scenario, this would load actual true labels
    return [1, 1, 1, 0, 0]
