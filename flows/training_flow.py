from pathlib import Path
import subprocess
from typing import List, Dict, Optional
from prefect import flow, task
#from prefect_docker import DockerContainer

# ────────────────────────────────────────────────────────────────────────────
# Paths — resolve **project root**, then point to data/ and mlruns/
#   flows/
#     training_flow.py   ← this file   (__file__)
#   ↑ project root       ← parent of flows/
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # …/MLO_exercises
DATA_DIR     = PROJECT_ROOT / "data"
MLRUNS_DIR   = PROJECT_ROOT / "mlruns"

# For Windows paths, Docker just needs simple strings
DATA_DIR   = str(DATA_DIR)
MLRUNS_DIR = str(MLRUNS_DIR)


class DockerContainer:
    def __init__(
        self,
        image: str,
        command: str,
        volumes: Optional[List[str]] = None,
        env:    Optional[Dict[str,str]] = None,
        image_pull_policy: str = "IF_NOT_PRESENT",
        stream_output: bool = True,
        name: Optional[str] = None,
    ):
        self.image  = image
        self.command = command
        self.volumes = volumes or []
        self.env     = env or {}
        self.name    = name or image

    def run(self):
        # build base docker command
        cmd = ["docker", "run", "--rm", "--name", self.name]
        # mount volumes
        for v in self.volumes:
            cmd += ["-v", v]
        # pass through env vars
        for k, v in self.env.items():
            cmd += ["-e", f"{k}={v}"]
        # image and actual command
        cmd.append(self.image)
        cmd += self.command.split()

        # execute, streaming stdout/stderr
        subprocess.run(cmd, check=True)

# ────────────────────────────────────────────────────────────────────────────
# 1️⃣  Data-quality tests  (image built from ./pre_training_tests)
# ────────────────────────────────────────────────────────────────────────────
@task
def run_data_tests() -> None:
    DockerContainer(
        image="pre-training-tests-image:latest",         # build tag: docker build -t pre-training-tests-image ./pre_training_tests
        command="python main.py",
        volumes=[
            f"{DATA_DIR}:/app/data:ro",                  # dataset, read-only
        ],
        image_pull_policy="IF_NOT_PRESENT",
        stream_output=True,
        name="Run data-quality tests",
    ).run()

# ────────────────────────────────────────────────────────────────────────────
# 2️⃣  Model training  (image built from ./model/train)
# ────────────────────────────────────────────────────────────────────────────
@task
def train_model() -> None:
    DockerContainer(
        image="model-train-image:latest",                # docker build -t model-train-image ./model/train
        command="python train.py",
        volumes=[
            f"{DATA_DIR}:/app/data:ro",                  # dataset
            f"{MLRUNS_DIR}:/app/mlruns",                 # where MLflow will log runs + artefacts
        ],
        env={
            "MLFLOW_TRACKING_URI": "file:/app/mlruns",
        },
        image_pull_policy="IF_NOT_PRESENT",
        stream_output=True,
        name="Train model",
    ).run()

# ────────────────────────────────────────────────────────────────────────────
# 3️⃣  Robustness validation  (image built from ./model/validate)
# ────────────────────────────────────────────────────────────────────────────
@task
def validate_model_robustness() -> None:
    DockerContainer(
        image="model-validate-image:latest",             # docker build -t model-validate-image ./model/validate
        command="python validate_robustness.py",
        volumes=[
            f"{MLRUNS_DIR}:/app/mlruns:ro",              # artefacts only – read-only is fine
        ],
        env={
            "MLFLOW_TRACKING_URI": "file:/app/mlruns",
        },
        image_pull_policy="IF_NOT_PRESENT",
        stream_output=True,
        name="Validate robustness",
    ).run()

# ────────────────────────────────────────────────────────────────────────────
# Prefect Flow orchestration
# ────────────────────────────────────────────────────────────────────────────
@flow(name="Training Workflow")
def training_flow() -> None:
    run_data_tests()
    train_model()
    validate_model_robustness()

if __name__ == "__main__":
    training_flow()
