#!/usr/bin/env bash
#SBATCH --job-name=univision_E3
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/univision_E3_%j.out
#SBATCH --error=logs/univision_E3_%j.err

set -euo pipefail
cd "${UNIVISION_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="$PWD/.venv/bin/python"
DATA="$PWD/data/MeViS_release"
TOKEN_CACHE="$PWD/outputs/clip_tokens_32.pt"
OUT="$PWD/outputs/E3_fastvit_t8_2080ti"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p logs outputs "$OUT"

echo "E3 job=${SLURM_JOB_ID:-unknown} host=$(hostname) date=$(date)"
nvidia-smi
"$PYTHON" -m pytest -q tests/test_e3.py

if [[ ! -s "$TOKEN_CACHE" ]]; then
    echo "Generating word-level CLIP cache; this may take several minutes and ~1GB disk"
    "$PYTHON" -u -m univision.e3_text_cache \
        --data "$DATA" --splits train valid_u \
        --max-length 32 --batch-size 16 --out "$TOKEN_CACHE"
fi

# First verify real FastViT, the token cache, and the new cross-attention.
if [[ ! -f outputs/E3_REAL_PREFLIGHT_OK ]]; then
    "$PYTHON" -u -m univision.e3_train \
      --data "$DATA" --token-cache "$TOKEN_CACHE" \
      --encoder fastvit_t8.apple_in1k \
      --image-size 128 --clip-length 2 --batch-size 1 --workers 0 \
      --width 48 --epochs 1 --limit-train 8 --limit-val 4 \
      --freeze-encoder --seed 43 --out outputs/E3_REAL_PREFLIGHT
    touch outputs/E3_REAL_PREFLIGHT_OK
fi

RESUME_ARGS=()
if [[ -s "$OUT/last.pt" ]]; then
    echo "Resuming E3: $OUT/last.pt"
    RESUME_ARGS=(--resume "$OUT/last.pt")
fi

"$PYTHON" -u -m univision.e3_train \
  --data "$DATA" --token-cache "$TOKEN_CACHE" \
  --encoder fastvit_t8.apple_in1k \
  --image-size 224 --clip-length 3 --batch-size 1 \
  --workers 2 --width 48 --epochs 12 --grad-accum 2 \
  --presence-weight 0.5 --corruption-probability 0.3 \
  --lr 0.0002 --freeze-encoder --amp --seed 43 \
  --out "$OUT" "${RESUME_ARGS[@]}"
echo "E3 job done at $(date)"
