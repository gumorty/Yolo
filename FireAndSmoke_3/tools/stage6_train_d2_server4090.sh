#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-0,1}"
BATCH="${BATCH:-24}"
IMGSZ="${IMGSZ:-960}"
WORKERS="${WORKERS:-12}"
EPOCHS_PHASE1="${EPOCHS_PHASE1:-3}"
EPOCHS_PHASE2="${EPOCHS_PHASE2:-12}"

GENERATED_DIR="$ROOT/generated"
GENERATED_DATA="$GENERATED_DIR/stage6_mixed_3cls_union_server.yaml"
mkdir -p "$GENERATED_DIR"

cat > "$GENERATED_DATA" <<YAML
train:
  - $ROOT/datasets/stage4_full_tile_sensors3/train/images
  - $ROOT/datasets/stage6_sources/verified_empty_fp_round01/train/images
  - $ROOT/datasets/stage6_sources/verified_empty_fp_consensus_round02/train/images
val:
  - $ROOT/datasets/stage4_full_tile_sensors3/val/images
  - $ROOT/datasets/stage6_sources/verified_empty_fp_round01/val/images
  - $ROOT/datasets/stage6_sources/verified_empty_fp_consensus_round02/val/images
test:
  - $ROOT/datasets/stage4_full_tile_sensors3/test/images
nc: 3
names:
  - fire
  - other
  - smoke
YAML

DATA="${DATA:-$GENERATED_DATA}"
MODEL_YAML="${MODEL_YAML:-$ROOT/models/yolov8m-p2-fire-smoke-3cls.yaml}"
PROJECT="${PROJECT:-$ROOT/runs_stage6_server4090}"

find_weight() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PRETRAINED="${PRETRAINED:-}"
if [[ -z "$PRETRAINED" ]]; then
  PRETRAINED="$(find_weight \
    "$ROOT/models_stage4/s5_best.pt" \
    "$ROOT/../Yolo/models_stage4/s5_best.pt" \
    "$ROOT/../../Yolo/models_stage4/s5_best.pt" \
    "$ROOT/models_stage4/v4_best.pt" \
    "$ROOT/../Yolo/models_stage4/v4_best.pt" \
    "$ROOT/../../Yolo/models_stage4/v4_best.pt")" || {
      echo "Could not find S5/V4 pretrained weight. Set PRETRAINED=/path/to/best.pt" >&2
      exit 1
    }
fi

echo "ROOT=$ROOT"
echo "DATA=$DATA"
echo "MODEL_YAML=$MODEL_YAML"
echo "PRETRAINED=$PRETRAINED"
echo "DEVICE=$DEVICE BATCH=$BATCH IMGSZ=$IMGSZ"

python tools/train_yolo_api.py \
  --model "$MODEL_YAML" \
  --data "$DATA" \
  --project "$PROJECT" \
  --name stage6_d2_server4090_phase1_freeze_s5_union_i${IMGSZ} \
  --pretrained "$PRETRAINED" \
  --fallback-pretrained "" \
  --epochs "$EPOCHS_PHASE1" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --patience 3 \
  --mosaic 0.05 \
  --mixup 0.0 \
  --copy-paste 0.0 \
  --close-mosaic 1 \
  --auto-augment none \
  --erasing 0.0 \
  --optimizer AdamW \
  --lr0 0.0001 \
  --lrf 0.5 \
  --weight-decay 0.0005 \
  --warmup-epochs 0.0 \
  --warmup-bias-lr 0.0001 \
  --freeze 10 \
  --amp true \
  --save-period 1

PHASE1_BEST="$PROJECT/stage6_d2_server4090_phase1_freeze_s5_union_i${IMGSZ}/weights/best.pt"
if [[ ! -f "$PHASE1_BEST" ]]; then
  echo "Missing phase1 best weight: $PHASE1_BEST" >&2
  exit 1
fi

python tools/train_yolo_api.py \
  --model "$PHASE1_BEST" \
  --data "$DATA" \
  --project "$PROJECT" \
  --name stage6_d2_server4090_phase2_unfreeze_s5_union_i${IMGSZ} \
  --fallback-pretrained "" \
  --epochs "$EPOCHS_PHASE2" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --patience 5 \
  --mosaic 0.03 \
  --mixup 0.0 \
  --copy-paste 0.0 \
  --close-mosaic 3 \
  --auto-augment none \
  --erasing 0.0 \
  --optimizer AdamW \
  --lr0 0.00006 \
  --lrf 0.2 \
  --weight-decay 0.0005 \
  --warmup-epochs 0.0 \
  --warmup-bias-lr 0.00006 \
  --amp true \
  --save-period 2

echo "D2 server training complete."
echo "Best weight: $PROJECT/stage6_d2_server4090_phase2_unfreeze_s5_union_i${IMGSZ}/weights/best.pt"
