import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prefect import flow, task
import subprocess

from model.train_model import train_model
from model.validate_robustness import validate_robustness

@task
def run_data_tests():
    result = subprocess.run([sys.executable, "pre_training_tests/main.py"], check=True)
    return result.returncode

@task
def train_model():
    train()

@task
def validate_model_robustness():
    validate_robustness()


@flow(name="Training Workflow")
def training_flow():
    run_data_tests()
    train_model()
    validate_model_robustness()

if __name__ == "__main__":
    training_flow()
