#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$HOME/gu}"
PROJECT="${PROJECT:-$BASE/projects/FireAndSmoke_3}"
ROOT="${ROOT:-$PROJECT}"
mkdir -p "$ROOT/reports"

for name in strict_sensors_3cls strict_sensors_2cls self_fire_pic_2cls_standalone; do
  YAML="$ROOT/datasets/$name/data.yaml"
  if [[ -f "$YAML" ]]; then
    python "$PROJECT/tools/dataset_audit.py" \
      --data "$YAML" \
      --out "$ROOT/reports/$name.audit.json"
  else
    echo "Skip missing $YAML"
  fi
done
