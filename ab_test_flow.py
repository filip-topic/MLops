import pandas as pd
import numpy as np
import mlflow
import time
from mlflow import MlflowClient
import os
from sklearn.metrics import accuracy_score
import yaml
import os
from pathlib import Path
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

#mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
mlflow.set_tracking_uri("file:./mlruns")
print("-------MLflow tracking URI set to:-------------------", mlflow.get_tracking_uri())


@task
def load_config(path="model/train/config.yaml"):
    time.sleep(0.1)
    with open(path, "r") as f:
        return yaml.safe_load(f)

@task
def load_test_data(n_test=2000):
    time.sleep(0.1)
    """
    Loads the last n_test rows of the dataset as the held-out test set.
    """
    df = pd.read_csv('data/Womens Clothing E-Commerce Reviews.csv')
    test_df = df.tail(n_test).reset_index(drop=True)
    return test_df

@task
def get_latest_two_run_ids():
    time.sleep(0.1)
    """
    Returns the run IDs of the latest two model runs from MLflow.
    """
    #mlflow.set_tracking_uri("file:./mlruns")
    #mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("recommendation-models")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=2)
    if len(runs) < 2:
        raise RuntimeError("Not enough model runs found in MLflow for A/B testing. Train at least two models.")
    return [run.info.run_id for run in runs]

'''@task
def load_model(run_id):
    """
    Loads a model from MLflow given a run ID.
    """
    print(f"Current working directory: {os.getcwd()}")
    try:
        return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    except Exception as e:
        raise RuntimeError(f"Failed to load model for run_id={run_id}: {e}")'''

def load_model(run_id):
    time.sleep(0.1)
    """
    Loads a model from MLflow given a run ID, fixing the artifact URI path if needed.
    """
    try:
        # Step 1: Get experiment ID for this run
        client = MlflowClient()
        run = client.get_run(run_id)
        experiment_id = run.info.experiment_id

        # Step 2: Build correct path to meta.yaml
        meta_path = Path("mlruns") / experiment_id / run_id / "meta.yaml"
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)

        # Step 3: Replace container path with local path
        recorded_uri = meta["artifact_uri"]
        corrected_uri = recorded_uri.replace(
            "file:///app/mlruns",
            Path("mlruns").resolve().as_uri()
        )

        model_path = corrected_uri + "/model"
        print(f"Corrected model path: {model_path}")

        model = mlflow.sklearn.load_model(model_path)
        print(f"STEPS OF THE MODEL: {model.named_steps}")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model for run_id={run_id}: {e}")

@task
def ab_split(df):
    time.sleep(0.1)
    """
    Splits the test set evenly and reproducibly by index parity (even/odd rows).
    """
    a_df = df.iloc[::2].reset_index(drop=True)
    b_df = df.iloc[1::2].reset_index(drop=True)
    return a_df, b_df

@task
def run_predictions(model, df, features):
    time.sleep(0.1)
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in test data: {missing_cols}")
    X = df[features].dropna()
    if X.empty:
        raise ValueError("No valid rows in test data after dropping missing values for required features.")
    y = df.loc[X.index, "Recommended IND"]
    preds = model.predict(X)
    return y, preds

@flow(name="AB Test Flow", task_runner=ConcurrentTaskRunner())
def ab_test_flow(n_test: int = 2000):
    """
    A/B test flow for comparing two model versions on held-out data.
    - Loads last two model versions from MLflow.
    - Splits test set evenly by index parity.
    - Compares accuracy for each model.
    - Documents how to handle multiple concurrent/subsequent A/B tests.
    """
    config = load_config()
    features = [col for col in config["data"]["required_columns"] if col != "Recommended IND"]
    test_df = load_test_data(n_test)
    run_ids = get_latest_two_run_ids()
    model_a = load_model(run_ids[0])
    model_b = load_model(run_ids[1])
    a_df, b_df = ab_split(test_df)
    y_a, preds_a = run_predictions(model_a, a_df, features)
    y_b, preds_b = run_predictions(model_b, b_df, features)
    acc_a = accuracy_score(y_a, preds_a)
    acc_b = accuracy_score(y_b, preds_b)
    print(f"Model A (run_id={run_ids[0]}) accuracy: {acc_a:.4f}")
    print(f"Model B (run_id={run_ids[1]}) accuracy: {acc_b:.4f}")
    # To handle multiple A/B tests, use unique test IDs and log results with metadata (e.g., timestamp, test_id).
    # This allows tracking and separation of concurrent/subsequent tests in logs or a database.

if __name__ == "__main__":
    ab_test_flow() 