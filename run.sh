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
docker build -t "${STEP_IMG_TRAIN}"       ./model/train
docker build -t "${STEP_IMG_VALIDATE}"    ./model/validate

echo "✔ Step images built."

# ─────────────────────────────────────────────────────────────
# OPTIONAL: build the minimal orchestrator image
# Commented out by default; see README for details.
# ─────────────────────────────────────────────────────────────
echo "▶ Building orchestrator image…"
docker build -t "${ORCH_IMG}" .
echo "✔ Orchestrator image built."

echo
# ─────────────────────────────────────────────────────────────
# Run the orchestrator container, mounting Docker socket + data
# ─────────────────────────────────────────────────────────────
echo "▶ Launching orchestrator container…"
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \      # let it spawn other containers
  -v "$(pwd)":/workspace \                             # give it the project
  -w /workspace/flows \
  training-orchestrator \
  python training_flow.py