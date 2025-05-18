import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score

def test_model_accuracy(true_labels, model_predictions):
    """Test that model accuracy meets minimum threshold."""
    min_accuracy = 0.7  # Minimum acceptable accuracy
    accuracy = accuracy_score(true_labels, model_predictions)
    assert accuracy >= min_accuracy, f"Model accuracy {accuracy:.4f} is below threshold {min_accuracy}"

def test_model_precision(true_labels, model_predictions):
    """Test that model precision meets minimum threshold."""
    min_precision = 0.7  # Minimum acceptable precision
    precision = precision_score(true_labels, model_predictions, zero_division=0)
    assert precision >= min_precision, f"Model precision {precision:.4f} is below threshold {min_precision}"

def test_model_recall(true_labels, model_predictions):
    """Test that model recall meets minimum threshold."""
    min_recall = 0.6  # Minimum acceptable recall
    recall = recall_score(true_labels, model_predictions, zero_division=0)
    assert recall >= min_recall, f"Model recall {recall:.4f} is below threshold {min_recall}"

def test_prediction_shape(model_predictions):
    """Test that predictions have the expected shape."""
    assert len(model_predictions) > 0, "No predictions returned"
    assert all(isinstance(p, (int, float)) for p in model_predictions), "Predictions should be numeric"
