#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/gu/projects/FireAndSmoke_3}"
DATA="${DATA:-$PROJECT/datasets/stage4_full_tile_sensors3/data.yaml}"
OUT="${OUT:-$PROJECT/reports/stage4_full_tile_sensors3.audit.json}"

if [[ ! -f "$DATA" ]]; then
  echo "Missing data yaml: $DATA"
  exit 1
fi

python "$PROJECT/tools/dataset_audit.py" --data "$DATA" --out "$OUT"
echo "Audit written to $OUT"
