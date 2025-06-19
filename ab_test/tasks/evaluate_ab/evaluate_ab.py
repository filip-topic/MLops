import argparse, json, pathlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-ids-json", default="/outputs/run_ids.json")
    p.add_argument("--output-dir", default="/outputs")
    args = p.parse_args()

    with open(args.run_ids_json) as f:
        run_ids = json.load(f)

    metrics = {}
    model_configs = {}
    
    for rid in run_ids:
        metrics_path = pathlib.Path(args.output_dir) / f"metrics_{rid}.json"
        with open(metrics_path) as mf:
            result = json.load(mf)
            metrics[rid] = result["accuracy"]
            model_configs[rid] = {
                "params": result.get("model_params", {}),
                "features": result.get("features", [])
            }

    best_run = max(metrics, key=metrics.get)
    
    # Create comprehensive evaluation report
    evaluation_report = {
        "accuracies": metrics,
        "best_run": best_run,
        "best_accuracy": metrics[best_run],
        "model_configs": model_configs,
        "summary": {
            "total_models": len(run_ids),
            "accuracy_range": {
                "min": min(metrics.values()),
                "max": max(metrics.values()),
                "avg": sum(metrics.values()) / len(metrics.values())
            }
        }
    }
    
    # Save detailed evaluation report
    out_path = pathlib.Path(args.output_dir) / "evaluation_report.json"
    with open(out_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print(json.dumps(evaluation_report, indent=2))

if __name__ == "__main__":
    main()