#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$HOME/gu}"
PROJECT="${PROJECT:-$BASE/projects/FireAndSmoke_3}"
ROOT="${ROOT:-$PROJECT}"
EPOCHS="${EPOCHS:-120}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-960}"
DEVICE="${DEVICE:-0}"
DATA="$ROOT/datasets/strict_sensors_3cls/data.yaml"
MODEL="$PROJECT/models/yolov8m-p2-fire-smoke-3cls.yaml"
PRETRAINED="${PRETRAINED:-$BASE/weights/sensors_yolov8m_3cls_100_best.pt}"
PROJECT_PRETRAINED="$PROJECT/weights/sensors_yolov8m_3cls_100_best.pt"

if [[ ! -f "$DATA" ]]; then
  echo "Missing data yaml: $DATA"
  exit 1
fi
if [[ ! -f "$PRETRAINED" ]]; then
  if [[ -f "$PROJECT_PRETRAINED" ]]; then
    echo "Use project bundled pretrained weight: $PROJECT_PRETRAINED"
    PRETRAINED="$PROJECT_PRETRAINED"
  else
    echo "Missing pretrained weight: $PRETRAINED"
    echo "Fallback to yolov8m.pt"
    PRETRAINED="yolov8m.pt"
  fi
fi

mkdir -p "$BASE/runs" "$BASE/logs"

python "$PROJECT/tools/train_yolo_api.py" \
  --model "$MODEL" \
  --pretrained "$PRETRAINED" \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --project "$BASE/runs" \
  --name "stage3_yolov8m_p2_sensors3_e${EPOCHS}" \
  --patience 35 \
  --close-mosaic 20 \
  --mosaic 0.7 \
  --mixup 0.05 \
  --copy-paste 0.05 \
  --workers 8 \
  --amp false
