from pathlib import Path
import subprocess
from prefect import flow, task
from typing import List, Dict, Optional
import json, pathlib

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "data"
MLRUNS_DIR = PROJECT_ROOT.parent / "mlruns"
ARTIFACT_DIR = PROJECT_ROOT.parent / "artifacts" / "ab_test"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

class DockerContainer:
    def __init__(self, image: str, command: str, volumes: List[str], env: Dict[str, str] | None = None, name: str | None = None):
        self.image = image; self.command = command; self.volumes = volumes; self.env = env or {}; self.name = name or image.replace(":", "-")
    def run(self):
        cmd = ["docker", "run", "--rm", "--name", self.name]
        for v in self.volumes: cmd += ["-v", v]
        for k, v in self.env.items(): cmd += ["-e", f"{k}={v}"]
        cmd.append(self.image); cmd += self.command.split()
        print(" →", " ".join(cmd)); subprocess.run(cmd, check=True)

@task
def step_load_data(n_test:int):
    DockerContainer(
        image="ab-load_test_data:latest",
        command=f"python load_test_data.py --n-test {n_test} --output-dir /outputs",
        volumes=[
            f"{DATA_DIR}:/app/data:ro",
            f"{ARTIFACT_DIR}:/outputs",
        ],
        name="ab-load_data"
    ).run()

@task
def step_get_runs():
    DockerContainer(
        image="ab-get_latest_two_run_ids:latest",
        command="python get_latest_two_run_ids.py --output-dir /outputs",
        volumes=[
            f"{MLRUNS_DIR}:/app/mlruns:ro",
            f"{ARTIFACT_DIR}:/outputs",
        ],
        name="ab-get_runs"
    ).run()

@task
def step_predictions_per_run(run_id: str):
    DockerContainer(
        image="ab-run_predictions:latest",
        command=f"python run_predictions.py --run-id {run_id} --output-dir /outputs",
        volumes=[
            f"{MLRUNS_DIR}:/app/mlruns:ro",
            f"{ARTIFACT_DIR}:/outputs",
        ],
        name=f"ab-preds-{run_id}"
    ).run()

@task
def step_eval():
    DockerContainer(
        image="ab-evaluate_ab:latest",
        command="python evaluate_ab.py --output-dir /outputs",
        volumes=[f"{ARTIFACT_DIR}:/outputs"],
        name="ab-eval"
    ).run()

@flow(name="AB‑Test‑Flow")
def ab_test_flow(n_test:int=2000):
    step_load_data(n_test)
    step_get_runs()

    run_ids_path = ARTIFACT_DIR / "run_ids.json"
    run_ids = json.loads(run_ids_path.read_text())
    for rid in run_ids:
        step_predictions_per_run(rid)

    step_eval()

if __name__ == "__main__":
    ab_test_flow()