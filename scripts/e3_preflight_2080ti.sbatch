#!/usr/bin/env bash
#SBATCH --job-name=univision_E3_check
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/univision_E3_check_%j.out
#SBATCH --error=logs/univision_E3_check_%j.err
set -euo pipefail
cd "${UNIVISION_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="$PWD/.venv/bin/python"
DATA="$PWD/data/MeViS_release"
TOKEN_CACHE="$PWD/outputs/clip_tokens_32.pt"
export PYTHONUNBUFFERED=1
mkdir -p logs outputs
nvidia-smi
"$PYTHON" -m pytest -q tests/test_e3.py
if [[ ! -s "$TOKEN_CACHE" ]]; then
    "$PYTHON" -u -m univision.e3_text_cache --data "$DATA" --splits train valid_u --max-length 32 --batch-size 16 --out "$TOKEN_CACHE"
fi
"$PYTHON" -u -m univision.e3_train \
  --data "$DATA" --token-cache "$TOKEN_CACHE" \
  --encoder fastvit_t8.apple_in1k \
  --image-size 128 --clip-length 2 --batch-size 1 --workers 0 \
  --width 48 --epochs 1 --limit-train 8 --limit-val 4 \
  --freeze-encoder --seed 43 --out outputs/E3_REAL_PREFLIGHT

touch outputs/E3_REAL_PREFLIGHT_OK
echo "E3 real-FastViT and CLIP-token preflight PASSED"
