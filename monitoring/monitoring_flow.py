from pathlib import Path
import subprocess
from prefect import flow, task
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "data"
ARTIFACT_DIR = PROJECT_ROOT.parent / "artifacts" / "monitoring"
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
def task_load_data(n_test: int):
    DockerContainer(
        image="monitoring-load-data:latest",
        command=f"--n-test {n_test} --output-dir /outputs",
        volumes=[
            f"{DATA_DIR}:/app/data:ro",
            f"{ARTIFACT_DIR}:/outputs"
        ],
        name="mon-load-data"
    ).run()

@task
def task_compute_kl():
    DockerContainer(
        image="monitoring-kl-div:latest",
        command="--train-path /outputs/train.parquet --test-path /outputs/test.parquet --output-dir /outputs",
        volumes=[f"{ARTIFACT_DIR}:/outputs"],
        name="mon-kl-div"
    ).run()

@flow(name="Monitoring‑Drift‑Flow")
def monitoring_flow(n_test: int = 2000):
    task_load_data(n_test)
    task_compute_kl()

if __name__ == "__main__":
    monitoring_flow()