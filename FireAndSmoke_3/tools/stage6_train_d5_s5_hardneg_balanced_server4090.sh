#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-0,1}"
BATCH="${BATCH:-24}"
IMGSZ="${IMGSZ:-960}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS:-8}"
HARD_REPEAT="${HARD_REPEAT:-4}"

PROJECT="${PROJECT:-$ROOT/runs_stage6_server4090}"
RUN_NAME="${RUN_NAME:-stage6_d5_s5_hardneg_balanced_x${HARD_REPEAT}_i${IMGSZ}}"
GENERATED_DIR="$ROOT/generated"
GENERATED_DATA="$GENERATED_DIR/stage6_d5_s5_hardneg_balanced_x${HARD_REPEAT}_server.yaml"
mkdir -p "$GENERATED_DIR"

SOURCE_WEIGHT="${SOURCE_WEIGHT:-$ROOT/models_stage4/s5_best.pt}"
if [[ ! -f "$SOURCE_WEIGHT" ]]; then
  echo "Missing S5 source weight: $SOURCE_WEIGHT" >&2
  echo "Set SOURCE_WEIGHT=/path/to/s5_best.pt if needed." >&2
  exit 1
fi

{
  echo "train:"
  echo "  - $ROOT/datasets/stage4_full_tile_sensors3/train/images"
  for _ in $(seq 1 "$HARD_REPEAT"); do
    echo "  - $ROOT/datasets/stage6_sources/verified_empty_fp_round01/train/images"
    echo "  - $ROOT/datasets/stage6_sources/verified_empty_fp_consensus_round02/train/images"
  done
  echo "val:"
  echo "  - $ROOT/datasets/stage4_full_tile_sensors3/val/images"
  echo "  - $ROOT/datasets/stage6_sources/verified_empty_fp_round01/val/images"
  echo "  - $ROOT/datasets/stage6_sources/verified_empty_fp_consensus_round02/val/images"
  echo "test:"
  echo "  - $ROOT/datasets/stage4_full_tile_sensors3/test/images"
  echo "nc: 3"
  echo "names:"
  echo "  - fire"
  echo "  - other"
  echo "  - smoke"
} > "$GENERATED_DATA"

echo "ROOT=$ROOT"
echo "SOURCE_WEIGHT=$SOURCE_WEIGHT"
echo "DATA=$GENERATED_DATA"
echo "PROJECT=$PROJECT"
echo "RUN_NAME=$RUN_NAME"
echo "DEVICE=$DEVICE BATCH=$BATCH IMGSZ=$IMGSZ EPOCHS=$EPOCHS HARD_REPEAT=$HARD_REPEAT"

python tools/train_yolo_api.py \
  --model "$SOURCE_WEIGHT" \
  --data "$GENERATED_DATA" \
  --project "$PROJECT" \
  --name "$RUN_NAME" \
  --fallback-pretrained "" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --patience 4 \
  --mosaic 0.0 \
  --mixup 0.0 \
  --copy-paste 0.0 \
  --close-mosaic 0 \
  --auto-augment none \
  --erasing 0.0 \
  --optimizer AdamW \
  --lr0 0.00003 \
  --lrf 0.2 \
  --weight-decay 0.0005 \
  --warmup-epochs 0.0 \
  --warmup-bias-lr 0.00003 \
  --freeze 10 \
  --amp true \
  --save-period 1

echo "D5 S5-balanced hard-negative calibration complete."
echo "Best weight: $PROJECT/$RUN_NAME/weights/best.pt"
