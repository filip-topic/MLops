import argparse, yaml, json, pathlib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="/app/config.yaml")
    p.add_argument("--output-dir", default="/outputs")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg))
    print(json.dumps(cfg, indent=2))

if __name__ == "__main__":
    main()