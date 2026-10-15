#!/bin/bash
#SBATCH --job-name=univision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=logs/univision_%j.out
# Adapt the account/partition/modules to your SLURM cluster.
set -euo pipefail
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate univision
cd "$SLURM_SUBMIT_DIR"
python -m univision.train \
    --data "$HOME/datasets/MeViS_release" \
    --text-cache outputs/clip_text.pt \
    --out outputs/fastvit_multimodal \
    --encoder fastvit_t8.apple_in1k --epochs 20 \
    --image-size 256 --batch-size 2 --clip-length 4 --workers 4 \
    --freeze-encoder --amp
