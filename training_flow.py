from pathlib import Path
import subprocess
from typing import List, Dict, Optional
from prefect import flow, task
import os
import yaml
#from prefect_docker import DockerContainer

# ────────────────────────────────────────────────────────────────────────────
# Paths — resolve **project root**, then point to data/ and mlruns/
#   flows/
#     training_flow.py   ← this file   (__file__)
#   ↑ project root       ← parent of flows/
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent   # …/MLO_exercises
DATA_DIR = PROJECT_ROOT.as_posix() + "/data"
MLRUNS_DIR = PROJECT_ROOT.as_posix() + "/mlruns"
TESTS_DIR = PROJECT_ROOT.as_posix() + "/pre_training_tests"

# For Windows paths, Docker just needs simple strings
DATA_DIR = str(DATA_DIR)
MLRUNS_DIR = str(MLRUNS_DIR)
TESTS_DIR = str(TESTS_DIR)

def load_config(path="./model/train/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_param(config, config_path, default=None):
    keys = config_path.split('.')
    cfg = config
    for k in keys:
        cfg = cfg.get(k, None)
        if cfg is None:
            return default
    return cfg


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
        self.image = image
        self.command = command
        self.volumes = volumes or []
        self.env = env or {}
        self.name = name or image

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
        print(cmd)
        subprocess.run(cmd, check=True)


# task for debugging
@task
def debug_listing():
    DockerContainer(
        image="pre-training-tests-image:latest",
        command="bash -lc 'ls -l /app/data'",
        volumes=[ f"{DATA_DIR}:/app/data:ro" ],
        name="debug-data-mount",
    ).run()


# ────────────────────────────────────────────────────────────────────────────
# 1️⃣  Data-quality tests  (image built from ./pre_training_tests)
# ────────────────────────────────────────────────────────────────────────────
@task
def run_data_tests() -> None:
    DockerContainer(
        image="pre-training-tests-image:latest",
        command="python main.py",
        # mount data read-only, but mount host pre_training_tests for output
        volumes=[
            f"{DATA_DIR}:/app/data:ro",
            f"{PROJECT_ROOT}/pre_training_tests:/app"      # ← here
        ],
        name="run-data-quality-tests",
    ).run()

# ────────────────────────────────────────────────────────────────────────────
# 2️⃣  Model training  (image built from ./model/train)
# ────────────────────────────────────────────────────────────────────────────
@task
def train_model(
    min_training_size: int = None,
    max_iter: int = None,
    random_state: int = 42
) -> None:
    DockerContainer(
        image="model-train-image:latest",
        command="python train.py",
        volumes=[
            f"{DATA_DIR}:/app/data:ro",
            f"{MLRUNS_DIR}:/app/mlruns",
            f"{TESTS_DIR}:/app/pre_training_tests"
        ],
        env={
            "MLFLOW_TRACKING_URI": "file:/app/mlruns",
            "MIN_TRAINING_SIZE": str(min_training_size),
            "MAX_ITER": str(max_iter),
            "RANDOM_STATE": str(random_state),
        },
        image_pull_policy="IF_NOT_PRESENT",
        stream_output=True,
        name="train-model",
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
        name="validate-robustness",
    ).run()

# ────────────────────────────────────────────────────────────────────────────
# Prefect Flow orchestration
# ────────────────────────────────────────────────────────────────────────────

config = load_config()

min_training_size = get_param(config, "training.min_training_size", 1000)
max_iter = get_param(config, "training.max_iter", 10)
random_state = get_param(config, "training.random_state", 42)

@flow(name="Training Workflow")
def training_flow(
    min_training_size: int = min_training_size,
    max_iter: int = max_iter,
    random_state: int = random_state
) -> None:
    run_data_tests()
    train_model(min_training_size, max_iter, random_state)
    validate_model_robustness()

if __name__ == "__main__":
    training_flow() #.serve.run()
