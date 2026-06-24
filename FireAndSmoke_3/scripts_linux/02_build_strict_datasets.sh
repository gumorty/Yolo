#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$HOME/gu}"
PROJECT="${PROJECT:-$BASE/projects/FireAndSmoke_3}"
ROOT="${ROOT:-$PROJECT}"
SENSORS3="${SENSORS3:-$BASE/datasets/raw_yolov8/yolov8}"
SENSORS2="${SENSORS2:-$BASE/datasets/binary_fire_smoke}"
SELF2="${SELF2:-$BASE/datasets/fire_pic/fire_dataset_all}"

mkdir -p "$ROOT/datasets" "$ROOT/reports"

ARGS=(--root "$ROOT" --sensors3 "$SENSORS3" --sensors2 "$SENSORS2")
if [[ -d "$SELF2" ]]; then
  ARGS+=(--self2 "$SELF2")
fi

python "$PROJECT/tools/build_strict_datasets.py" "${ARGS[@]}"
