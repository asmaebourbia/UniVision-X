#!/usr/bin/env bash
#SBATCH --job-name=univision_E2
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/univision_E2_%j.out
#SBATCH --error=logs/univision_E2_%j.err

set -euo pipefail

# Always run from the UniVision-X project (shared storage).
cd "${UNIVISION_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${PWD}/.venv/bin/python"
DATA="${PWD}/data/MeViS_release"
CACHE="${PWD}/outputs/clip_text.pt"
OUT="${PWD}/outputs/E2_fastvit_t8_2080ti"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: virtual environment missing: $PYTHON" >&2
    exit 1
fi

for split in train valid_u; do
    split_dir="$DATA/$split"
    if [[ ! -d "$split_dir/JPEGImages" || ! -f "$split_dir/mask_dict.json" ]]; then
        echo "ERROR: missing MeViS files (JPEGImages/ or mask_dict.json): $split_dir" >&2
        exit 1
    fi
    if [[ ! -f "$split_dir/meta_expressions.json" ]]; then
        if [[ -f "$split_dir/meta_expressions_v2.json" ]]; then
            echo "Creating metadata link for $split"
            ln -s meta_expressions_v2.json "$split_dir/meta_expressions.json"
        else
            echo "ERROR: missing meta_expressions.json in $split_dir" >&2
            exit 1
        fi
    fi
done

mkdir -p "$OUT" outputs

echo "Starting job ${SLURM_JOB_ID:-unknown} on $(hostname) at $(date)"
nvidia-smi
"$PYTHON" -c 'import torch, timm, transformers; print("Torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), "timm:", timm.__version__, "transformers:", transformers.__version__)'

if [[ ! -s "$CACHE" ]]; then
    echo "CLIP cache missing. Computing it before training (requires CLIP weights in cache or Internet)."
    "$PYTHON" -u -m univision.text_cache \
        --data "$DATA" \
        --splits train valid_u \
        --model openai/clip-vit-base-patch32 \
        --batch-size 16 \
        --out "$CACHE"
fi

# Exercise the *real* FastViT (not the toy encoder) on four MeViS samples
# before reserving the remainder of the job for the full experiment.
if [[ ! -f "${PWD}/outputs/FASTVIT_PREFLIGHT_OK" ]]; then
    echo "Running a short real-FastViT preflight on 4 train / 2 validation examples"
    "$PYTHON" -u -m univision.train \
        --data "$DATA" --text-cache "$CACHE" \
        --encoder fastvit_t8.apple_in1k \
        --image-size 224 --clip-length 3 --batch-size 1 \
        --workers 2 --width 48 --epochs 1 \
        --limit-train 4 --limit-val 2 --freeze-encoder --amp \
        --seed 43 --out outputs/FASTVIT_PREFLIGHT
    touch outputs/FASTVIT_PREFLIGHT_OK
fi

RESUME_ARGS=()
if [[ -s "$OUT/last.pt" ]]; then
    echo "Resuming previous training checkpoint: $OUT/last.pt"
    RESUME_ARGS=(--resume "$OUT/last.pt")
fi

echo "Launching FastViT-T8 + CLIP + ConvGRU training"
"$PYTHON" -u -m univision.train \
    --data "$DATA" \
    --text-cache "$CACHE" \
    --encoder fastvit_t8.apple_in1k \
    --image-size 224 \
    --clip-length 3 \
    --batch-size 1 \
    --workers 2 \
    --width 48 \
    --epochs 20 \
    --freeze-encoder \
    --amp \
    --seed 43 \
    --out "$OUT" \
    "${RESUME_ARGS[@]}"

echo "Completed at $(date)"
