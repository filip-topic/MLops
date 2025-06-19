import argparse, pandas as pd, pathlib, json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/app/data/Womens Clothing E-Commerce Reviews.csv")
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--output-dir", default="/outputs")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    test_df = df.tail(args.n_test).reset_index(drop=True)

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(out / "test.parquet", index=False)

    meta = {"rows": len(test_df), "parquet": str(out / "test.parquet")}
    (out / "meta_test.json").write_text(json.dumps(meta))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()