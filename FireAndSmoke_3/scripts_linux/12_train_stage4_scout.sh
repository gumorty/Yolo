#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$HOME/gu}"
PROJECT="${PROJECT:-$BASE/projects/FireAndSmoke_3}"
RUN_ROOT="${RUN_ROOT:-$BASE/stage4}"
DATA="${DATA:-$PROJECT/datasets/stage4_full_tile_sensors3/data.yaml}"
MODEL="${MODEL:-$PROJECT/models/yolov8m-p2-fire-smoke-3cls.yaml}"
PRETRAINED="${PRETRAINED:-$PROJECT/weights/sensors_yolov8m_3cls_100_best.pt}"
NAME="${NAME:-stage4_scout_p2_full_tile_e20}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-960}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"

if [[ ! -f "$DATA" ]]; then
  echo "Missing data yaml: $DATA. Run scripts_linux/11_build_highres_tiles.sh first, or set DATA."
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Missing model yaml: $MODEL"
  exit 1
fi
if [[ ! -f "$PRETRAINED" ]]; then
  echo "Missing pretrained weight: $PRETRAINED; fallback to yolov8m.pt"
  PRETRAINED="yolov8m.pt"
fi

mkdir -p "$RUN_ROOT/runs"

python "$PROJECT/tools/train_yolo_api.py" \
  --model "$MODEL" \
  --pretrained "$PRETRAINED" \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --project "$RUN_ROOT/runs" \
  --name "$NAME" \
  --patience 8 \
  --close-mosaic 5 \
  --mosaic 0.20 \
  --mixup 0.0 \
  --copy-paste 0.0 \
  --erasing 0.0 \
  --auto-augment none \
  --save-period 5 \
  --workers "$WORKERS" \
  --amp true
