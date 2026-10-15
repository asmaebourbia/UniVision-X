#!/usr/bin/env bash
#SBATCH --job-name=univision_eval
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/univision_eval_%j.out
#SBATCH --error=logs/univision_eval_%j.err

set -euo pipefail
cd "${UNIVISION_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="$PWD/.venv/bin/python"
CHECKPOINT="$PWD/outputs/E2_fastvit_t8_2080ti/best.pt"
TEXT_CACHE="$PWD/outputs/clip_text.pt"
DATA="$PWD/data/MeViS_release"
OUT="$PWD/outputs/E2_eval_clean"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

[[ -x "$PYTHON" ]] || { echo "Python absent: $PYTHON" >&2; exit 1; }
[[ -s "$CHECKPOINT" ]] || { echo "Checkpoint absent: $CHECKPOINT" >&2; exit 1; }
[[ -s "$TEXT_CACHE" ]] || { echo "Cache CLIP absent: $TEXT_CACHE" >&2; exit 1; }
mkdir -p "$OUT"
echo "Evaluation UniVision-X job ${SLURM_JOB_ID:-unknown} @ $(date)"
nvidia-smi
"$PYTHON" -u -m univision.evaluate \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --text-cache "$TEXT_CACHE" \
  --split valid_u \
  --chunk-size 2 \
  --amp \
  --out "$OUT"
echo "Evaluation terminee @ $(date) ; rapport: $OUT/summary.json"
