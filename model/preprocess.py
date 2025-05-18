# model/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop rows with missing target or key structured features
    df = df.dropna(subset=["Recommended IND", "Age", "Rating", "Division Name", "Department Name", "Class Name"])

    # Define features and target
    features = ["Age", "Rating", "Positive Feedback Count", "Division Name", "Department Name", "Class Name"]
    target = "Recommended IND"

    X = df[features]
    y = df[target]

    # Identify categorical columns
    categorical_cols = ["Division Name", "Department Name", "Class Name"]
    numeric_cols = ["Age", "Rating", "Positive Feedback Count"]

    # Preprocessing: One-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    return X, y, preprocessor
