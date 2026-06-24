#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/gu/projects/FireAndSmoke_3}"
CONFIG="${CONFIG:-$PROJECT/configs/stage4_dataset_mix_template.yaml}"
OUT="${OUT:-$PROJECT/datasets/stage4_mixed_3cls}"
OVERWRITE="${OVERWRITE:-0}"

ARGS=(--config "$CONFIG" --out "$OUT")
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python "$PROJECT/tools/build_stage4_mixed_dataset.py" "${ARGS[@]}"
