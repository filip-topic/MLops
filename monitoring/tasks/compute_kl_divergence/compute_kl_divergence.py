import argparse, pandas as pd, numpy as np, json, pathlib
from scipy.stats import entropy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", default="/outputs/train.parquet")
    p.add_argument("--test-path", default="/outputs/test.parquet")
    p.add_argument("--feature", default="Age")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--output-dir", default="/outputs")
    args = p.parse_args()

    train_df = pd.read_parquet(args.train_path)
    test_df = pd.read_parquet(args.test_path)

    train_hist, bin_edges = np.histogram(train_df[args.feature].dropna(), bins=args.bins, density=True)
    test_hist, _ = np.histogram(test_df[args.feature].dropna(), bins=bin_edges, density=True)

    train_hist += 1e-8
    test_hist += 1e-8
    kl_div = float(entropy(test_hist, train_hist))

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"feature": args.feature, "kl_divergence": kl_div}
    (out / "drift_metrics.json").write_text(json.dumps(metrics))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()