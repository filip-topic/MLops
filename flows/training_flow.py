# flows/training_flow.py

from prefect import flow, task
import subprocess
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

@task
def run_data_tests():
    result = subprocess.run(["python", "pre_training_tests/main.py"], check=True)
    return result.returncode

@task
def train_model():
    df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")

    df = df.dropna(subset=["Recommended IND", "Age", "Rating", "Division Name", "Department Name", "Class Name"])
    features = ["Age", "Rating", "Positive Feedback Count", "Division Name", "Department Name", "Class Name"]
    target = "Recommended IND"

    X = df[features]
    y = df[target]

    numeric_cols = ["Age", "Rating", "Positive Feedback Count"]
    categorical_cols = ["Division Name", "Department Name", "Class Name"]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pipeline.fit(X_train, y_train)

    model_path = "model/logistic_model.joblib"
    joblib.dump(pipeline, model_path)

    return model_path

@flow(name="Training Workflow")
def training_flow():
    run_data_tests()
    model_path = train_model()
    print(f"✅ Model saved to: {model_path}")

if __name__ == "__main__":
    training_flow()
