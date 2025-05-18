# model/train_model.py

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

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

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")  # store locally
    mlflow.set_experiment("recommendation-models")

    with mlflow.start_run() as run:
        signature = infer_signature(X_train, pipeline.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name="recommendation_model"
        )

        mlflow.log_params({"model_type": "LogisticRegression"})

        # 🔽 Log environment metadata
        mlflow.log_artifact("model/requirements.txt")

        print(f"✅ Model logged to MLflow with Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train_model()
