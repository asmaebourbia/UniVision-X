#!/usr/bin/env bash
#SBATCH --job-name=uvx_text_diag
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/uvx_text_diag_%j.out
#SBATCH --error=logs/uvx_text_diag_%j.err
set -euo pipefail
cd "${UNIVISION_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
.venv/bin/python diagnose_text_grounding.py \
    --checkpoint outputs/E2_fastvit_t8_2080ti/best.pt \
    --data data/MeViS_release \
    --text-cache outputs/clip_text.pt \
    --split valid_u \
    --limit 100 \
    --chunk-size 2 \
    --amp \
    --out outputs/E2_text_diagnostic
