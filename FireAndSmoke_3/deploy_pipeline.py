#!/usr/bin/env python3
"""
Deploy and start Stage4 training pipeline on remote server.
Steps:
1. Upload holdout eval script
2. Upload E60 resume training script  
3. Upload pipeline wrapper script
4. Start pipeline in background with nohup
"""
import paramiko
import os
import time
import sys

HOST = os.getenv('YOLO_REMOTE_HOST')
PORT = int(os.getenv('YOLO_REMOTE_PORT', '22'))
USER = os.getenv('YOLO_REMOTE_USER')
PASS = os.getenv('YOLO_REMOTE_PASSWORD')
if not all([HOST, USER, PASS]):
    raise RuntimeError(
        'Set YOLO_REMOTE_HOST, YOLO_REMOTE_PORT, YOLO_REMOTE_USER, '
        'and YOLO_REMOTE_PASSWORD before running deployment.'
    )
REMOTE_BASE = '/home/uav/gu/stage4'

# ============================================================
# Script 1: Holdout Evaluation
# ============================================================
HOLDOUT_EVAL_SCRIPT = r'''#!/usr/bin/env python3
"""Evaluate E20 best.pt on public hard holdout dataset."""
import os
os.environ['MPLBACKEND'] = 'Agg'

import json
import time
from datetime import datetime

print(f"[{datetime.now()}] Holdout evaluation starting...")

from ultralytics import YOLO

WEIGHTS = '/home/uav/gu/stage4/runs/stage4_scout_p2_fasdd_dfire_tile_e20/weights/best.pt'
DATA_YAML = '/home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_eval/public_hard_holdout/data.yaml'
PROJECT = '/home/uav/gu/stage4/eval_public_holdout'

# Evaluate best.pt
print(f"[{datetime.now()}] Loading model: {WEIGHTS}")
model = YOLO(WEIGHTS)

print(f"[{datetime.now()}] Running validation on holdout...")
results = model.val(
    data=DATA_YAML,
    split='test',
    save_json=True,
    save_conf=True,
    plots=False,
    device='0',
    name='holdout_e20_best_noplots',
    project=PROJECT,
    conf=0.25,
    iou=0.6,
)

# Print and save results
print("=" * 60)
print("HOLDOUT EVALUATION RESULTS - E20 best.pt")
print("=" * 60)
print(f"mAP50:    {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall:    {results.box.mr:.4f}")

# Per-class results
print("\nPer-class results:")
for i, name in enumerate(results.names.values()):
    print(f"  {name}: AP50={results.box.ap50[i]:.4f}, AP={results.box.ap[i]:.4f}")

# Save results to JSON
result_data = {
    'timestamp': datetime.now().isoformat(),
    'weights': WEIGHTS,
    'data': DATA_YAML,
    'mAP50': float(results.box.map50),
    'mAP50-95': float(results.box.map),
    'precision': float(results.box.mp),
    'recall': float(results.box.mr),
    'per_class': {
        name: {'AP50': float(results.box.ap50[i]), 'AP': float(results.box.ap[i])}
        for i, name in enumerate(results.names.values())
    }
}

result_path = os.path.join(PROJECT, 'holdout_e20_best_results.json')
with open(result_path, 'w') as f:
    json.dump(result_data, f, indent=2)
print(f"\nResults saved to: {result_path}")
print(f"[{datetime.now()}] Holdout evaluation complete!")

# Also evaluate epoch5, epoch10, epoch15 for progression analysis
for epoch_num in [5, 10, 15]:
    epoch_weights = f'/home/uav/gu/stage4/runs/stage4_scout_p2_fasdd_dfire_tile_e20/weights/epoch{epoch_num}.pt'
    if not os.path.exists(epoch_weights):
        print(f"\n[{datetime.now()}] Skipping epoch{epoch_num} - file not found")
        continue

    print(f"\n[{datetime.now()}] Evaluating epoch{epoch_num} on holdout...")
    model_ep = YOLO(epoch_weights)
    results_ep = model_ep.val(
        data=DATA_YAML,
        split='test',
        save_json=False,
        save_conf=True,
        plots=False,
        device='0',
        name=f'holdout_epoch{epoch_num}_noplots',
        project=PROJECT,
        conf=0.25,
        iou=0.6,
    )

    print(f"  Epoch{epoch_num}: mAP50={results_ep.box.map50:.4f}, mAP50-95={results_ep.box.map:.4f}, "
          f"P={results_ep.box.mp:.4f}, R={results_ep.box.mr:.4f}")

    # Save each epoch result
    ep_data = {
        'timestamp': datetime.now().isoformat(),
        'weights': epoch_weights,
        'data': DATA_YAML,
        'mAP50': float(results_ep.box.map50),
        'mAP50-95': float(results_ep.box.map),
        'precision': float(results_ep.box.mp),
        'recall': float(results_ep.box.mr),
        'per_class': {
            name: {'AP50': float(results_ep.box.ap50[i]), 'AP': float(results_ep.box.ap[i])}
            for i, name in enumerate(results_ep.names.values())
        }
    }
    ep_path = os.path.join(PROJECT, f'holdout_epoch{epoch_num}_results.json')
    with open(ep_path, 'w') as f:
        json.dump(ep_data, f, indent=2)
    print(f"  Results saved to: {ep_path}")

print(f"\n[{datetime.now()}] ALL holdout evaluations complete!")
'''

# ============================================================
# Script 2: E60 Resume Training
# ============================================================
E60_RESUME_SCRIPT = r'''#!/usr/bin/env python3
"""Resume Stage4 E60 training from last.pt checkpoint."""
import os
os.environ['MPLBACKEND'] = 'Agg'

from datetime import datetime

print(f"[{datetime.now()}] E60 Resume training starting...")

from ultralytics import YOLO

LAST_PT = '/home/uav/gu/stage4/runs/stage4_p2_fasdd_dfire_tile_e60_from_e20best/weights/last.pt'

print(f"[{datetime.now()}] Loading checkpoint: {LAST_PT}")
model = YOLO(LAST_PT)

print(f"[{datetime.now()}] Starting resume training (plots=False to avoid font crash)...")
model.train(resume=True, plots=False)

print(f"[{datetime.now()}] E60 Resume training complete!")
'''

# ============================================================
# Script 3: 120-epoch Long Training (prepared but not auto-started)
# ============================================================
E120_TRAIN_SCRIPT = r'''#!/usr/bin/env python3
"""Stage4 formal 120-epoch long training from E20 best.pt.
This script is PREPARED but NOT auto-started. 
Run manually after reviewing E60 results.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

from datetime import datetime

print(f"[{datetime.now()}] Stage4 120-epoch training starting...")

from ultralytics import YOLO

# Load E20 best as starting point
E20_BEST = '/home/uav/gu/stage4/runs/stage4_scout_p2_fasdd_dfire_tile_e20/weights/best.pt'
DATA_YAML = '/home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_full_tile_sensors3/data.yaml'

print(f"[{datetime.now()}] Loading base model: {E20_BEST}")
model = YOLO(E20_BEST)

print(f"[{datetime.now()}] Starting 120-epoch training...")
model.train(
    data=DATA_YAML,
    epochs=120,
    imgsz=1280,
    device='0,1',
    batch=6,
    lr0=0.002,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5,
    warmup_momentum=0.5,
    optimizer='AdamW',
    weight_decay=0.0005,
    mosaic=0.1,
    mixup=0.02,
    copy_paste=0.1,
    degrees=10,
    translate=0.1,
    scale=0.3,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.2,
    patience=30,
    save_period=5,
    plots=False,
    val=True,
    project='/home/uav/gu/stage4/runs',
    name='stage4_p2_fasdd_dfire_tile_e120_from_e20best',
    exist_ok=False,
    pretrained=False,
)

print(f"[{datetime.now()}] Stage4 120-epoch training complete!")
'''

# ============================================================
# Script 4: Pipeline Wrapper (holdout eval -> E60 resume)
# ============================================================
PIPELINE_SCRIPT = r'''#!/bin/bash
# Stage4 Pipeline: Holdout Eval -> E60 Resume Training
# Runs in background via nohup

set -e

LOG_DIR="/home/uav/gu/stage4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Stage4 Pipeline Started: $(date)"
echo "=============================================="

# Step 1: Run Holdout Evaluation (~10-30 min)
echo ""
echo "[Step 1/2] Starting Holdout Evaluation..."
echo "[$(date)] Holdout eval starting" >> ${LOG_DIR}/pipeline_status.log
cd /home/uav/gu
MPLBACKEND=Agg python3 ${LOG_DIR}/run_holdout_eval.py 2>&1 | tee -a ${LOG_DIR}/holdout_eval_full.log
HOLDOUT_EXIT=$?
if [ $HOLDOUT_EXIT -eq 0 ]; then
    echo "[$(date)] Holdout eval completed successfully" >> ${LOG_DIR}/pipeline_status.log
    echo "[Step 1/2] Holdout Evaluation COMPLETE!"
else
    echo "[$(date)] Holdout eval FAILED with exit code $HOLDOUT_EXIT" >> ${LOG_DIR}/pipeline_status.log
    echo "[Step 1/2] Holdout Evaluation FAILED! Exit code: $HOLDOUT_EXIT"
    echo "Continuing with training anyway..."
fi

# Step 2: Resume E60 Training (~1-2 days)
echo ""
echo "[Step 2/2] Starting E60 Resume Training..."
echo "[$(date)] E60 resume training starting" >> ${LOG_DIR}/pipeline_status.log
cd /home/uav/gu
MPLBACKEND=Agg python3 ${LOG_DIR}/run_e60_resume.py 2>&1 | tee -a ${LOG_DIR}/stage4_e60_resume.log
TRAIN_EXIT=$?
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "[$(date)] E60 resume training completed successfully" >> ${LOG_DIR}/pipeline_status.log
    echo "[Step 2/2] E60 Resume Training COMPLETE!"
else
    echo "[$(date)] E60 resume training FAILED with exit code $TRAIN_EXIT" >> ${LOG_DIR}/pipeline_status.log
    echo "[Step 2/2] E60 Resume Training FAILED! Exit code: $TRAIN_EXIT"
fi

echo ""
echo "=============================================="
echo "Stage4 Pipeline Finished: $(date)"
echo "Holdout eval exit: $HOLDOUT_EXIT"
echo "E60 training exit: $TRAIN_EXIT"
echo "=============================================="
echo "[$(date)] Pipeline finished. Holdout=$HOLDOUT_EXIT, E60=$TRAIN_EXIT" >> ${LOG_DIR}/pipeline_status.log
'''

def main():
    print("Connecting to server...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS)
    
    # Use SFTP to upload scripts
    sftp = ssh.open_sftp()
    
    scripts = {
        f'{REMOTE_BASE}/run_holdout_eval.py': HOLDOUT_EVAL_SCRIPT,
        f'{REMOTE_BASE}/run_e60_resume.py': E60_RESUME_SCRIPT,
        f'{REMOTE_BASE}/run_e120_train.py': E120_TRAIN_SCRIPT,
        f'{REMOTE_BASE}/run_pipeline.sh': PIPELINE_SCRIPT,
    }
    
    for remote_path, content in scripts.items():
        print(f"Uploading {remote_path}...")
        with sftp.open(remote_path, 'w') as f:
            f.write(content)
        print(f"  -> Uploaded successfully")
    
    sftp.close()
    
    # Set permissions and start pipeline
    commands = [
        # Make scripts executable
        f'chmod +x {REMOTE_BASE}/run_pipeline.sh {REMOTE_BASE}/run_holdout_eval.py {REMOTE_BASE}/run_e60_resume.py {REMOTE_BASE}/run_e120_train.py',
        # Verify files exist
        f'echo "=== Uploaded files ===" && ls -la {REMOTE_BASE}/run_*.py {REMOTE_BASE}/run_*.sh',
        # Create initial status log
        f'echo "[$(date)] Pipeline scripts deployed, ready to start" > {REMOTE_BASE}/pipeline_status.log',
        # Kill any existing pipeline processes (just in case)
        'pkill -f run_pipeline.sh 2>/dev/null; pkill -f run_holdout_eval 2>/dev/null; pkill -f run_e60_resume 2>/dev/null; echo "Cleaned up old processes"',
        # Start pipeline in background with nohup
        f'cd /home/uav/gu && nohup bash {REMOTE_BASE}/run_pipeline.sh > {REMOTE_BASE}/pipeline_nohup.log 2>&1 & echo "Pipeline PID: $!"',
        # Wait a moment and verify it started
        'sleep 3 && echo "=== Running processes ===" && ps aux | grep -E "run_pipeline|run_holdout|run_e60" | grep -v grep',
        # Show initial log
        f'echo "=== Initial pipeline log ===" && head -20 {REMOTE_BASE}/pipeline_nohup.log 2>/dev/null || echo "Log not yet available"',
    ]
    
    for cmd in commands:
        print(f"\n>>> {cmd[:100]}...")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode()
        err = stderr.read().decode()
        if out: print(out.strip())
        if err and 'no process found' not in err.lower() and 'cleaned up' not in err.lower(): 
            print(f"STDERR: {err.strip()}")
    
    ssh.close()
    print("\n=== Deployment complete! Pipeline is running in background ===")

if __name__ == '__main__':
    main()
