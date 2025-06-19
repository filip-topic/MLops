import argparse, json, pathlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-ids-json", default="/outputs/run_ids.json")
    p.add_argument("--output-dir", default="/outputs")
    args = p.parse_args()

    with open(args.run_ids_json) as f:
        run_ids = json.load(f)

    metrics = {}
    for rid in run_ids:
        metrics_path = pathlib.Path(args.output_dir) / f"metrics_{rid}.json"
        with open(metrics_path) as mf:
            metrics[rid] = json.load(mf)["accuracy"]

    best = max(metrics, key=metrics.get)
    print(json.dumps({"accuracies": metrics, "best_run": best}, indent=2))

if __name__ == "__main__":
    main()