import argparse
import pandas as pd
import pathlib
import json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", default="/app/data/Womens Clothing E-Commerce Reviews.csv")
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--output-dir", default="/outputs")
    args = p.parse_args()

    df = pd.read_csv(args.input_path)
    test_df = df.tail(args.n_test)
    train_df = df.iloc[:-args.n_test]

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out / "train.parquet", index=False)
    test_df.to_parquet(out / "test.parquet", index=False)

    # store meta so downstream knows where artefacts live
    meta = {
        "train_path": str(out / "train.parquet"),
        "test_path": str(out / "test.parquet"),
        "n_test": args.n_test,
    }
    (out / "meta_load_data.json").write_text(json.dumps(meta))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()