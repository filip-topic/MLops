import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow


def validate_robustness():
    print("Running robustness validation...")

    # Load the latest model run
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name("recommendation-models")
    if experiment is None:
        raise ValueError("Experiment 'recommendation-models' not found. Has the model been logged properly?")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment.name}' (ID: {experiment.experiment_id})")

    run_id = runs[0].info.run_id
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")
    df = df.dropna(subset=["Recommended IND", "Age", "Rating", "Division Name", "Department Name", "Class Name"])
    
    features = ["Age", "Rating", "Positive Feedback Count", "Division Name", "Department Name", "Class Name"]
    X = df[features]
    y = df["Recommended IND"]

    # Create perturbed version of numeric columns
    numeric_cols = ["Age", "Rating", "Positive Feedback Count"]
    X_perturbed = X.copy()
    for col in numeric_cols:
        X_perturbed[col] = X[col] * (1 + np.random.normal(0, 0.05, size=X.shape[0]))  # ±5% noise

    # Predict probabilities
    probs_original = model.predict_proba(X)[:, 1]
    probs_perturbed = model.predict_proba(X_perturbed)[:, 1]

    # Measure RMSE of change
    drift = np.sqrt(mean_squared_error(probs_original, probs_perturbed))
    print(f"Probability drift under noise: {drift:.4f}")

    threshold = 0.1
    if drift > threshold:
        print("Model fails robustness test — too sensitive to small input changes.")
    else:
        print("Model passes robustness expectation.")