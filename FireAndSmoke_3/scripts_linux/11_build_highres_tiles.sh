#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/gu/projects/FireAndSmoke_3}"
DATA="${DATA:-$PROJECT/datasets/strict_sensors_3cls/data.yaml}"
OUT="${OUT:-$PROJECT/datasets/stage4_full_tile_sensors3}"
TILE_SIZE="${TILE_SIZE:-960}"
OVERLAP="${OVERLAP:-0.25}"
MIN_VISIBILITY="${MIN_VISIBILITY:-0.35}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-0.10}"
COPY_FULL="${COPY_FULL:-1}"
OVERWRITE="${OVERWRITE:-0}"

ARGS=(
  --data "$DATA"
  --out "$OUT"
  --tile-size "$TILE_SIZE"
  --overlap "$OVERLAP"
  --min-visibility "$MIN_VISIBILITY"
  --negative-ratio "$NEGATIVE_RATIO"
  --positive-classes fire smoke
)
if [[ "$COPY_FULL" == "1" ]]; then
  ARGS+=(--copy-full)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python "$PROJECT/tools/build_highres_tile_dataset.py" "${ARGS[@]}"
