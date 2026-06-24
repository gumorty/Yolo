#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-0,1}"
WORKERS="${WORKERS:-12}"
IMGSZ="${IMGSZ:-960}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
PROJECT="${PROJECT:-$ROOT/runs_stage7_baselines}"
DATA="${DATA:-$ROOT/datasets/stage4_full_tile_sensors3/data.yaml}"
MODELS="${MODELS:-yolov8n.pt yolov8s.pt yolov8m.pt}"

if [[ ! -f "$DATA" ]]; then
  echo "Missing data yaml: $DATA" >&2
  exit 1
fi

mkdir -p "$PROJECT"

for MODEL in $MODELS; do
  STEM="${MODEL%.pt}"
  RUN_NAME="${RUN_NAME_PREFIX:-stage7}_${STEM}_stage4_i${IMGSZ}_e${EPOCHS}"
  case "$MODEL" in
    *yolov8n*) BATCH="${BATCH_N:-64}" ;;
    *yolov8s*) BATCH="${BATCH_S:-48}" ;;
    *) BATCH="${BATCH_M:-32}" ;;
  esac

  echo "============================================================"
  echo "MODEL=$MODEL"
  echo "RUN_NAME=$RUN_NAME"
  echo "DATA=$DATA"
  echo "PROJECT=$PROJECT"
  echo "DEVICE=$DEVICE BATCH=$BATCH IMGSZ=$IMGSZ EPOCHS=$EPOCHS PATIENCE=$PATIENCE"
  echo "============================================================"

  if [[ -d "$PROJECT/$RUN_NAME" ]]; then
    echo "Skip existing run directory: $PROJECT/$RUN_NAME"
    continue
  fi

  python tools/train_yolo_api.py \
    --model "$MODEL" \
    --data "$DATA" \
    --project "$PROJECT" \
    --name "$RUN_NAME" \
    --fallback-pretrained "" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz "$IMGSZ" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --patience "$PATIENCE" \
    --mosaic 0.7 \
    --mixup 0.05 \
    --copy-paste 0.05 \
    --close-mosaic 20 \
    --optimizer AdamW \
    --lr0 0.001 \
    --lrf 0.01 \
    --weight-decay 0.0005 \
    --amp true \
    --save-period 10 \
    --seed 0 \
    --deterministic true
done

echo "Stage7 YOLOv8 baseline suite complete."
