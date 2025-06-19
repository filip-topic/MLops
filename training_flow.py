from pathlib import Path
import subprocess
from typing import List, Dict, Optional
from prefect import flow, task
import os
import yaml
import mlflow
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

def load_config(path="./config.yaml"):
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
            f"{TESTS_DIR}:/app/pre_training_tests",
            f"{PROJECT_ROOT}/config.yaml:/app/config.yaml:ro"
        ],
        env={
            "MLFLOW_TRACKING_URI": "file:/app/mlruns",
            "MIN_TRAINING_SIZE": str(min_training_size),
            "MAX_ITER": str(max_iter),
            "RANDOM_STATE": str(random_state),
            "MLFLOW_PARENT_RUN_ID": os.environ.get("MLFLOW_PARENT_RUN_ID"),
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



@flow(name="Training Workflow")
def training_flow() -> None:
    
    config = load_config()
    min_training_size = get_param(config, "training.min_training_size", 1000)
    max_iter = get_param(config, "training.max_iter", 10)
    random_state = get_param(config, "training.random_state", 42)
    
    FLOWRUNS_DIR = PROJECT_ROOT.as_posix() + "/flow_runs"
    mlflow.set_tracking_uri(f"file:{FLOWRUNS_DIR}")
    mlflow.set_experiment("workflow-executions")
    with mlflow.start_run(run_name="training_workflow") as run:
        parent_run_id = run.info.run_id
        # Log workflow-level parameters
        mlflow.log_param("min_training_size", min_training_size)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)
        # Log config file as artifact
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)
        # Step 1: Data quality tests
        try:
            run_data_tests()
            mlflow.log_param("data_tests", "success")
        except Exception as e:
            mlflow.log_param("data_tests", f"failed: {e}")
            raise
        # Step 2: Model training (child run in Docker)
        try:
            train_model(min_training_size=min_training_size, 
                        max_iter=max_iter, 
                        random_state=random_state)
            mlflow.log_param("model_training", "success")
        except Exception as e:
            mlflow.log_param("model_training", f"failed: {e}")
            raise
        # Step 3: Robustness validation
        try:
            validate_model_robustness()
            mlflow.log_param("robustness_validation", "success")
        except Exception as e:
            mlflow.log_param("robustness_validation", f"failed: {e}")
            raise

if __name__ == "__main__":
    training_flow() 
