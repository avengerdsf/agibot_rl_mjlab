#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_FILE="/home/hxl228/.robot_rl_moniter/users/chen/repos/agibot_rl_mjlab/logs/rsl_rl/agibot_x1_velocity/2026-06-27_14-30-40/model_6900.pt"
OUTPUT_DIR="logs/freeze_rollout_$(date +%Y%m%d_%H%M%S)"
NUM_ENVS=2048
NUM_STEPS=2000
TASK_ID="AgiBot-X1-Flat"
CONDA_ENV="agibot"
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
