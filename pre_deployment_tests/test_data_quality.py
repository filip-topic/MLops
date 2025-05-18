import pytest
import pandas as pd
import numpy as np

def test_missing_values(sample_data):
    """Test that critical columns don't have too many missing values."""
    critical_columns = ['Review Text', 'Title', 'Rating']
    threshold = 0.4  # 40% threshold for missing values
    
    for col in critical_columns:
        missing_ratio = sample_data[col].isna().mean()
        assert missing_ratio < threshold, f"Column {col} has {missing_ratio*100:.2f}% missing values (threshold: {threshold*100}%)"

def test_data_types(sample_data):
    """Test that columns have the expected data types."""
    type_checks = {
        'Age': np.integer,
        'Rating': np.integer,
        'Recommended IND': np.integer,
        'Positive Feedback Count': np.integer,
        'Review Text': object,
        'Title': object
    }
    
    for col, expected_type in type_checks.items():
        assert isinstance(sample_data[col].iloc[0], (expected_type, type(None))), \
               f"Column {col} has unexpected type {type(sample_data[col].iloc[0])}. Expected {expected_type}"

def test_rating_range(sample_data):
    """Test that rating values are within the expected range."""
    assert sample_data['Rating'].between(1, 5).all(), "Rating values should be between 1 and 5"

def test_age_range(sample_data):
    """Test that age values are within a reasonable range."""
    assert sample_data['Age'].between(13, 115).all(), "Age values should be between 13 and 115"
