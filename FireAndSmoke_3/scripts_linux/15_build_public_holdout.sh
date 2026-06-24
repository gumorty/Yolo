#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/gu/projects/FireAndSmoke_3}"
DATA="${DATA:-$PROJECT/datasets/stage4_mixed_3cls/data.yaml}"
OUT="${OUT:-$PROJECT/datasets/stage4_eval/public_hard_holdout}"
MAX_IMAGES="${MAX_IMAGES:-500}"
OVERWRITE="${OVERWRITE:-0}"

ARGS=(--data "$DATA" --out "$OUT" --max-images "$MAX_IMAGES" --target-classes fire smoke)
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python "$PROJECT/tools/build_yolo_holdout.py" "${ARGS[@]}"
