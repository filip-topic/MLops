import argparse, json, pathlib, mlflow
from mlflow import MlflowClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", default="file:/app/mlruns")
    p.add_argument("--experiment-name", default="recommendation-models")
    p.add_argument("--output-dir", default="/outputs")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    runs = client.search_runs([experiment.experiment_id], order_by=["start_time desc"], max_results=2)
    run_ids = [run.info.run_id for run in runs]

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_ids.json").write_text(json.dumps(run_ids))
    print(json.dumps({"run_ids": run_ids}, indent=2))

if __name__ == "__main__":
    main()