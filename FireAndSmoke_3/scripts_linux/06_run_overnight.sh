#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$HOME/gu}"
PROJECT="${PROJECT:-$BASE/projects/FireAndSmoke_3}"
EPOCHS="${EPOCHS:-120}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-960}"
DEVICE="${DEVICE:-0}"
LOG_DIR="$BASE/logs"
STATUS="$LOG_DIR/stage3_status.log"
TRAIN_LOG="$LOG_DIR/stage3_train.log"

mkdir -p "$LOG_DIR"
echo "Stage3 training started at $(date)" | tee "$STATUS"
echo "Log: $TRAIN_LOG" | tee -a "$STATUS"

echo "== Train P2 Sensors 3-class ==" | tee -a "$STATUS"
EPOCHS="$EPOCHS" BATCH="$BATCH" IMGSZ="$IMGSZ" DEVICE="$DEVICE" \
  bash "$PROJECT/scripts_linux/04_train_p2_sensors3.sh" 2>&1 | tee -a "$TRAIN_LOG"
echo "DONE: stage3_yolov8m_p2_sensors3_e${EPOCHS}" | tee -a "$STATUS"

echo "== Train P2 Sensors 2-class ==" | tee -a "$STATUS"
EPOCHS="$EPOCHS" BATCH="$BATCH" IMGSZ="$IMGSZ" DEVICE="$DEVICE" \
  bash "$PROJECT/scripts_linux/05_train_p2_sensors2.sh" 2>&1 | tee -a "$TRAIN_LOG"
echo "DONE: stage3_yolov8m_p2_sensors2_e${EPOCHS}" | tee -a "$STATUS"

echo "Stage3 training finished at $(date)" | tee -a "$STATUS"
