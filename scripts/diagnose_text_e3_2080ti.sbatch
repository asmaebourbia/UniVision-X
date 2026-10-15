#!/usr/bin/env bash
#SBATCH --job-name=uvx_E3_text
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/uvx_E3_text_%j.out
#SBATCH --error=logs/uvx_E3_text_%j.err
set -euo pipefail
cd "${UNIVISION_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
.venv/bin/python -u diagnose_text_grounding_e3.py \
    --checkpoint outputs/E3_fastvit_t8_2080ti/best.pt \
    --data data/MeViS_release \
    --token-cache outputs/clip_tokens_32.pt \
    --split valid_u --seed 43 --limit 100 \
    --chunk-size 2 --presence-threshold 0.5 --amp \
    --out outputs/E3_text_diagnostic
