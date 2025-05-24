#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
STEP_IMG_DATA_TESTS="pre-training-tests-image:latest"
STEP_IMG_TRAIN="model-train-image:latest"
STEP_IMG_VALIDATE="model-validate-image:latest"
ORCH_IMG="training-orchestrator:latest"     # optional

echo "▶ Building step images…"

docker build -t "${STEP_IMG_DATA_TESTS}" ./pre_training_tests
docker build -t "${STEP_IMG_TRAIN}" ./model/train
docker build -t "${STEP_IMG_VALIDATE}" ./model/validate

echo "✔ Step images built."

pip install -r requirements.txt

python training_flow.py