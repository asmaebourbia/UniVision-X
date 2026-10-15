#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. Usage: bash scripts/robustness_eval.sh DATA_DIR TEXT_CACHE CHECKPOINT
DATA="${1:?data directory missing}"; TEXT="${2:?text cache missing}"; CKPT="${3:?checkpoint missing}"
for CONDITION in clean blur noise fog dark; do
  python -m univision.evaluate --data "$DATA" --text-cache "$TEXT" \
    --checkpoint "$CKPT" --corruption "$CONDITION" --chunk-size 4 \
    --out "outputs/robustness_${CONDITION}" --no-save-png
 done
