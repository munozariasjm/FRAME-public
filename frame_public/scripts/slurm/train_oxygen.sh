#!/bin/bash
#SBATCH --job-name=frame_o
#SBATCH --output=logs/train_oxygen_%j.out
#SBATCH --error=logs/train_oxygen_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=05:00:00

# Resolve paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_DIR="${REPO_ROOT}/data/oxygen"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/results/oxygen}"
PATTERN="O*.csv"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8192}"

mkdir -p "${SAVE_ROOT}" logs

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] Device: ${DEVICE}"

TAG="TRAIN"

python -u "${REPO_ROOT}/scripts/train.py" \
  --data_dir "${DATA_DIR}" \
  --file_pattern "${PATTERN}" \
  --save_root "${SAVE_ROOT}" \
  --exp_name "${TAG}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --target_cols "Energy ket,Rch"

# Find the run folder that matches this tag
EXP_DIR="$(ls -dt ${SAVE_ROOT}/*__${TAG} 2>/dev/null | head -n 1)"
if [[ -z "${EXP_DIR}" ]]; then
  echo "[ERROR] Could not locate experiment dir for tag ${TAG}"
  exit 1
fi
echo "[INFO] EXP_DIR=${EXP_DIR}"

python -u "${REPO_ROOT}/scripts/eval.py" \
  --exp_dir "${EXP_DIR}" \
  --fid_for_plots highest

echo "[DONE] ${EXP_DIR}"
