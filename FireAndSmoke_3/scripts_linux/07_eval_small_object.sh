#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$HOME/gu}"
PROJECT="${PROJECT:-$BASE/projects/FireAndSmoke_3}"
MODEL="${MODEL:-$BASE/runs/stage3_yolov8m_p2_sensors3_e120/weights/best.pt}"
DATA="${DATA:-$PROJECT/datasets/strict_sensors_3cls/data.yaml}"
IMGSZ="${IMGSZ:-960}"
CONF="${CONF:-0.20}"
IOU="${IOU:-0.50}"
OUT="${OUT:-$BASE/logs/small_object_eval.json}"

mkdir -p "$(dirname "$OUT")"
python "$PROJECT/tools/evaluate_small_objects.py" \
  --model "$MODEL" \
  --data "$DATA" \
  --imgsz "$IMGSZ" \
  --conf "$CONF" \
  --iou "$IOU" \
  --out "$OUT"
