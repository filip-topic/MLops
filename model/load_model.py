import mlflow.sklearn

def load_model_by_run_id(run_id: str):
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

if __name__ == "__main__":
    run_id = input("Enter MLflow Run ID: ").strip()
    model = load_model_by_run_id(run_id)
    print("Model loaded.")
