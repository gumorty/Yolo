"""
Ablation Study: YOLOv8m (no P2) Baseline Training
==================================================
Purpose: P2 ablation experiment - train standard YOLOv8m WITHOUT P2 detection head
         as the control group for proving P2's contribution to small fire/smoke detection.

Comparison Target: V4 best.pt (YOLOv8m-P2, imgsz=960)
  - V4 val: mAP50=0.844, mAP50-95=0.584
  - V4 holdout: mAP50=0.924, mAP50-95=0.681

Key Design Decisions:
- imgsz=960: SAME as V4 for fair P2 ablation (isolating P2 effect from image size effect)
- COCO pretrained yolov8m.pt as starting weights (standard practice for baseline)
- ALL hyperparameters matched to V4 training config EXCEPT model architecture
- This is a controlled experiment: only variable is P2 detection head presence
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

from datetime import datetime
from ultralytics import YOLO

# ============================================
# Configuration - Match V4 training exactly
# ============================================

# Model: Standard YOLOv8m (NO P2 head)
# Using yaml definition with 3 detection heads at strides 8/16/32
MODEL_YAML = '/home/uav/gu/ablation_p2/yolov8m_noP2_3cls.yaml'
# COCO pretrained weights for standard YOLOv8m
PRETRAINED = '/home/uav/yolov8m.pt'

# Dataset: SAME as V4
DATA_YAML = '/home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_full_tile_sensors3/data.yaml'

# Output: Separate directory, won't overwrite any existing results
PROJECT = '/home/uav/gu/ablation_p2/runs'
NAME = 'yolov8m_nop2_baseline_960'

print(f'[{datetime.now()}] ============================================')
print(f'[{datetime.now()}] P2 Ablation: YOLOv8m (no P2) Baseline Training')
print(f'[{datetime.now()}] ============================================')
print(f'[{datetime.now()}] Model YAML: {MODEL_YAML}')
print(f'[{datetime.now()}] Pretrained: {PRETRAINED}')
print(f'[{datetime.now()}] Data: {DATA_YAML}')
print(f'[{datetime.now()}] Output: {PROJECT}/{NAME}')
print()

# Load model from yaml (architecture only)
print(f'[{datetime.now()}] Loading model from yaml...')
model = YOLO(MODEL_YAML)
print(f'[{datetime.now()}] Model loaded. Parameters: {sum(p.numel() for p in model.model.parameters()):,}')

# Verify architecture
detect = model.model.model[-1]
print(f'[{datetime.now()}] Detection strides: {detect.stride.tolist()}')
print(f'[{datetime.now()}] Number of detection heads: {len(detect.cv2)} (expected: 3 for no-P2)')
print()

# Train with EXACT SAME hyperparameters as V4
# Reference: /home/uav/gu/stage4/runs/stage4_e60_v4_corrected/args.yaml
print(f'[{datetime.now()}] Starting training...')
results = model.train(
    data=DATA_YAML,
    epochs=100,           # More than V4's 60, but patience=20 will stop early
    imgsz=960,            # SAME as V4 - critical for fair ablation
    device='0,1',         # DDP 2x RTX 4090
    batch=16,             # SAME as V4

    # Optimizer - EXACTLY MATCH V4
    lr0=0.003,            # V4: 0.003
    lrf=0.01,             # V4: 0.01
    optimizer='MuSGD',    # V4: MuSGD
    momentum=0.937,       # V4: 0.937
    weight_decay=0.0005,  # V4: 0.0005
    cos_lr=True,          # V4: True
    warmup_epochs=3,      # V4: 3
    warmup_bias_lr=0.1,   # V4: 0.1
    warmup_momentum=0.8,  # V4: 0.8

    # Loss weights - SAME as V4 defaults
    box=7.5,              # V4: 7.5
    cls=0.5,              # V4: 0.5
    dfl=1.5,              # V4: 1.5

    # Augmentation - EXACTLY MATCH V4
    mosaic=0.15,          # V4: 0.15
    mixup=0.0,            # V4: 0.0
    copy_paste=0.0,       # V4: 0.0
    close_mosaic=10,      # V4: 10
    auto_augment='randaugment',  # V4: randaugment
    erasing=0.4,          # V4: 0.4
    hsv_h=0.015,          # V4: 0.015
    hsv_s=0.7,            # V4: 0.7
    hsv_v=0.4,            # V4: 0.4
    degrees=0.0,          # V4: 0.0
    translate=0.1,        # V4: 0.1
    scale=0.5,            # V4: 0.5
    fliplr=0.5,           # V4: 0.5

    # Training control
    patience=20,          # SAME as V4
    save=True,
    save_period=5,        # V4: 5
    plots=False,          # Avoid matplotlib crash
    amp=True,             # V4: True
    deterministic=True,   # V4: True

    # Other
    seed=20260614,        # Different seed from V4 for independence
    workers=8,            # V4: 8
    project=PROJECT,
    name=NAME,
    exist_ok=False,
    val=True,
    verbose=True,
    pretrained=True,      # Use COCO pretrained yolov8m weights
)

print(f'[{datetime.now()}] Training complete!')
