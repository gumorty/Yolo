#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WEIGHTS="${1:-}"
if [[ -z "$WEIGHTS" ]]; then
  WEIGHTS="$ROOT/runs_stage6_server4090/stage6_d2_server4090_phase2_unfreeze_s5_union_i960/weights/best.pt"
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS" >&2
  echo "Usage: bash tools/stage6_eval_model_server.sh /path/to/best.pt" >&2
  exit 1
fi

TAG="$(basename "$(dirname "$(dirname "$WEIGHTS")")")"
OUT_PUBLIC="$ROOT/reports/stage6_threshold_scan_${TAG}_public_hard.csv"
OUT_FALSE="$ROOT/reports/stage6_threshold_scan_${TAG}_false_alarm_v2.csv"

python tools/stage6_threshold_scan.py \
  --model "$WEIGHTS" \
  --data "$ROOT/datasets/stage4_eval/public_hard_holdout/data.yaml" \
  --split val \
  --imgsz 960 \
  --iou 0.50 \
  --out "$OUT_PUBLIC"

python tools/stage6_threshold_scan.py \
  --model "$WEIGHTS" \
  --data "$ROOT/datasets/stage6_eval/uav_false_alarm_v2/data.yaml" \
  --split val \
  --imgsz 960 \
  --iou 0.50 \
  --out "$OUT_FALSE"

echo "Evaluation complete."
echo "$OUT_PUBLIC"
echo "$OUT_FALSE"
