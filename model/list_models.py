import mlflow

def list_models():
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.MlflowClient()

    models = client.search_registered_models()
    for m in models:
        print(f"Model: {m.name}")
        for v in m.latest_versions:
            print(f"  - Version: {v.version}, Status: {v.status}, Run ID: {v.run_id}, URI: {v.source}")

if __name__ == "__main__":
    list_models()
