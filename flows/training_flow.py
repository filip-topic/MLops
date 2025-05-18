import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prefect import flow, task
import numpy as np
import mlflow.sklearn
import subprocess
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from model.train_model import train_model

@task
def run_data_tests():
    result = subprocess.run([sys.executable, "pre_training_tests/main.py"], check=True)
    return result.returncode

@task
def train():
    train_model()

@task
def validate_model_robustness():
    print("🔎 Running robustness validation...")

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
    print(f"🔧 Probability drift under noise: {drift:.4f}")

    threshold = 0.1
    if drift > threshold:
        print("⚠️  Model fails robustness test — too sensitive to small input changes.")
    else:
        print("✅ Model passes robustness expectation.")


@flow(name="Training Workflow")
def training_flow():
    run_data_tests()
    train()
    validate_model_robustness()

if __name__ == "__main__":
    training_flow()
