#!/usr/bin/env bash
#SBATCH --job-name=univision_E3_eval
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/univision_E3_eval_%j.out
#SBATCH --error=logs/univision_E3_eval_%j.err

set -euo pipefail
cd "${UNIVISION_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1
.venv/bin/python -u -m univision.e3_evaluate \
  --checkpoint outputs/E3_fastvit_t8_2080ti/best.pt \
  --token-cache outputs/clip_tokens_32.pt \
  --data data/MeViS_release --split valid_u \
  --chunk-size 2 --amp \
  --out outputs/E3_eval_clean
