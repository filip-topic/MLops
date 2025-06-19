import argparse, pandas as pd, mlflow, json, pathlib
from sklearn.metrics import accuracy_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--tracking-uri", default="file:/app/mlruns")
    p.add_argument("--test-path", default="/outputs/test.parquet")
    p.add_argument("--features-json", default="/outputs/config.json")
    p.add_argument("--output-dir", default="/outputs")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    with open(args.features_json) as f:
        cfg = json.load(f)
    features = [c for c in cfg["data"]["required_columns"] if c != "Recommended IND"]

    df = pd.read_parquet(args.test_path)
    model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model")

    y_true = df["Recommended IND"].values
    preds = model.predict(df[features])
    acc = float(accuracy_score(y_true, preds))

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"run_id": args.run_id, "accuracy": acc}
    (out / f"metrics_{args.run_id}.json").write_text(json.dumps(result))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()