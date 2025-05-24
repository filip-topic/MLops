# model/train_model.py

import os
import sys
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature





def load_config(path="./config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_model():

    try:
        config = load_config()

        df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")

        # Check required columns
        required_cols = config["data"]["required_columns"]
        if not all(col in df.columns for col in required_cols):
            missing = list(set(required_cols) - set(df.columns))
            msg = f"Missing required columns: {missing}"
            if config["training"]["fail_on_missing_columns"]:
                raise ValueError(msg)
            else:
                print(f"Warning: {msg}")

        # Drop rows missing required values
        df = df.dropna(subset=required_cols)

        # Size check
        if len(df) < config["training"]["min_training_size"]:
            raise ValueError(f"Training aborted: dataset too small ({len(df)} rows).")

        # --- Training Logic ---
        X = df[required_cols[:-1]]  # exclude target
        y = df["Recommended IND"]

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

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("recommendation-models")

        with mlflow.start_run() as run:
            pipeline.fit(X_train, y_train)

            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(3),
                registered_model_name="recommendation_model"
            )

            mlflow.log_params({"model_type": "LogisticRegression"})
            mlflow.log_artifact("requirements.txt")
            print(f"Model logged to MLflow with Run ID: {run.info.run_id}")

    except FileNotFoundError as fnf:
        print(f"File error: {fnf}")
    except ValueError as ve:
        print(f"Validation error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":

    ###############################################
    # this code makes sure that training proceeds only if training data is fine
    #################################################
    directory = "./pre_training_tests"
    missing_values_flag = os.path.join(directory, "missing_values_flag.ctl")
    distribution_flag = os.path.join(directory, "distribution_flag.ctl")
    # Check for missing_values_flag.ctl
    if not os.path.exists(missing_values_flag):
        print("Error: Too many missing values in the data")
        sys.exit(1)
    # Check for distribution_flag.ctl
    if not os.path.exists(distribution_flag):
        print("Error: Values in the data are outside the distribution. Something is wrong")
        sys.exit(1)
    # Proceed if both files exist
    print("Data distribution is fine.")
    print("Ammount of missing values is within reason.")
    print("Training can proceed.")

    # training the model
    train_model()
