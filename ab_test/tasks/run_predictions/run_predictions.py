import argparse, pandas as pd, mlflow, json, pathlib
from sklearn.metrics import accuracy_score
from mlflow import MlflowClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--tracking-uri", default="file:/app/mlruns")
    p.add_argument("--test-path", default="/outputs/test.parquet")
    p.add_argument("--output-dir", default="/outputs")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # Load model-specific parameters from MLflow run
    run = client.get_run(args.run_id)
    model_params = run.data.params
    
    # Load model-specific features from input example
    try:
        input_example_path = f"runs:/{args.run_id}/model/input_example.json"
        input_example = mlflow.artifacts.load_dict(input_example_path)
        features = list(input_example.columns)
    except Exception as e:
        print(f"Warning: Could not load input example for run {args.run_id}: {e}")
        # Fallback: use all columns except target
        df = pd.read_parquet(args.test_path)
        features = [col for col in df.columns if col != "Recommended IND"]

    df = pd.read_parquet(args.test_path)
    model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model")

    y_true = df["Recommended IND"].values
    preds = model.predict(df[features])
    acc = float(accuracy_score(y_true, preds))

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Save both metrics and model-specific config
    result = {
        "run_id": args.run_id, 
        "accuracy": acc,
        "model_params": dict(model_params),
        "features": features
    }
    (out / f"metrics_{args.run_id}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()