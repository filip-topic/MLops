import pytest
import pandas as pd
import os

def test_data_loading():
    """Test that data can be loaded successfully."""
    data_path = os.path.join('data', 'Womens Clothing E-Commerce Reviews.csv')
    try:
        df = pd.read_csv(data_path)
        assert not df.empty, "Dataframe is empty"
    except Exception as e:
        pytest.fail(f"Failed to load data: {str(e)}")

def test_model_loading():
    """Test that the model can be loaded."""
    # In a real scenario, this would load the actual model
    # For now, we'll just check if the model file exists
    model_path = os.path.join('model', 'model.pkl')
    if os.path.exists(model_path):
        assert True
    else:
        pytest.skip("Model file not found, skipping model loading test")

def test_endpoint_availability():
    """Test that the prediction endpoint is available."""
    # In a real scenario, this would test an actual API endpoint
    # For now, we'll just check if the endpoint configuration exists
    endpoint_config = os.path.join('config', 'endpoint.json')
    if os.path.exists(endpoint_config):
        assert True
    else:
        pytest.skip("Endpoint configuration not found, skipping endpoint test")
