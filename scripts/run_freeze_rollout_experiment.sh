#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_FILE="/path/to/checkpoint.pt"
OUTPUT_DIR="logs/freeze_rollout_$(date +%Y%m%d_%H%M%S)"
NUM_ENVS=4096
NUM_STEPS=2000
TASK_ID="AgiBot-X1-Flat"
CONDA_ENV="unitree_rl_mjlab"
FREEZE_MODE="zero"

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "Checkpoint file not found: ${CHECKPOINT_FILE}"
  echo "Edit CHECKPOINT_FILE at the top of $0."
  exit 2
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

mkdir -p "${OUTPUT_DIR}"

for FREEZE in none swing_hip_yaw swing_hip_roll swing_ankle_roll; do
  python scripts/eval_freeze_rollout.py "${TASK_ID}" \
    --checkpoint-file "${CHECKPOINT_FILE}" \
    --num-envs "${NUM_ENVS}" \
    --num-steps "${NUM_STEPS}" \
    --freeze "${FREEZE}" \
    --freeze-mode "${FREEZE_MODE}" \
    --output-file "${OUTPUT_DIR}/${FREEZE}.json"
done

echo "Freeze rollout results written to ${OUTPUT_DIR}"
