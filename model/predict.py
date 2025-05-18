# model/predict.py

import os
import pandas as pd
import joblib

def predict_from_csv(input_csv_path):
    # Load the saved model pipeline
    model_path = os.path.join("model", "logistic_model.pkl")
    model = joblib.load(model_path)

    # Load new data
    new_data = pd.read_csv(input_csv_path)

    # Ensure required columns are present
    required_columns = ["Age", "Rating", "Positive Feedback Count",
                        "Division Name", "Department Name", "Class Name"]
    
    if not all(col in new_data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in new_data.columns]
        raise ValueError(f"Missing columns in input data: {missing}")
    
    # Filter relevant columns and drop rows with missing values
    input_features = new_data[required_columns].dropna()

    # Predict
    predictions = model.predict(input_features)
    probabilities = model.predict_proba(input_features)[:, 1]  # Prob of recommending (label=1)

    # Combine with original data
    result_df = input_features.copy()
    result_df["Predicted_Recommended_IND"] = predictions
    result_df["Probability_Recommended"] = probabilities

    return result_df

if __name__ == "__main__":
    # Example usage
    input_path = os.path.join("data", "new_reviews.csv")  # Replace with your actual input file
    output_path = os.path.join("data", "predictions.csv")

    predictions_df = predict_from_csv(input_path)
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
