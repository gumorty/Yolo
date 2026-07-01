#!/usr/bin/env python3
"""Deploy and start Stage4 training pipeline v2 on remote server."""
import os
import paramiko
import time

HOST = os.getenv("YOLO_REMOTE_HOST")
PORT = int(os.getenv("YOLO_REMOTE_PORT", "22"))
USER = os.getenv("YOLO_REMOTE_USER")
PASS = os.getenv("YOLO_REMOTE_PASSWORD")
if not all([HOST, USER, PASS]):
    raise RuntimeError(
        "Set YOLO_REMOTE_HOST, YOLO_REMOTE_PORT, YOLO_REMOTE_USER, "
        "and YOLO_REMOTE_PASSWORD before running deployment."
    )
REMOTE_BASE = '/home/uav/gu/stage4'

HOLDOUT_SCRIPT = r"""#!/usr/bin/env python3
import os
os.environ["MPLBACKEND"] = "Agg"
import json
from datetime import datetime

print(f"[{datetime.now()}] Holdout evaluation starting...")
from ultralytics import YOLO

WEIGHTS = "/home/uav/gu/stage4/runs/stage4_scout_p2_fasdd_dfire_tile_e20/weights/best.pt"
DATA_YAML = "/home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_eval/public_hard_holdout/data.yaml"
PROJECT = "/home/uav/gu/stage4/eval_public_holdout"

print(f"[{datetime.now()}] Loading model: {WEIGHTS}")
model = YOLO(WEIGHTS)

print(f"[{datetime.now()}] Running validation on holdout...")
results = model.val(
    data=DATA_YAML, split="test", save_json=True, save_conf=True,
    plots=False, device="0", name="holdout_e20_best_v2",
    project=PROJECT, conf=0.25, iou=0.6,
)

print("=" * 60)
print("HOLDOUT EVALUATION RESULTS - E20 best.pt")
print("=" * 60)
print(f"mAP50:    {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall:    {results.box.mr:.4f}")

n_classes = len(results.box.ap50)
print(f"\nPer-class results ({n_classes} classes detected):")
for i in range(n_classes):
    name = results.names.get(i, f"class_{i}")
    print(f"  {name}: AP50={results.box.ap50[i]:.4f}, AP={results.box.ap[i]:.4f}")

result_data = {
    "timestamp": datetime.now().isoformat(),
    "weights": WEIGHTS,
    "data": DATA_YAML,
    "mAP50": float(results.box.map50),
    "mAP50-95": float(results.box.map),
    "precision": float(results.box.mp),
    "recall": float(results.box.mr),
    "per_class": {
        results.names.get(i, f"class_{i}"): {
            "AP50": float(results.box.ap50[i]),
            "AP": float(results.box.ap[i])
        }
        for i in range(n_classes)
    }
}

result_path = os.path.join(PROJECT, "holdout_e20_best_v2_results.json")
with open(result_path, "w") as f:
    json.dump(result_data, f, indent=2)
print(f"\nResults saved to: {result_path}")

# Evaluate epoch weights for progression analysis
for epoch_num in [5, 10, 15]:
    epoch_weights = f"/home/uav/gu/stage4/runs/stage4_scout_p2_fasdd_dfire_tile_e20/weights/epoch{epoch_num}.pt"
    if not os.path.exists(epoch_weights):
        print(f"\n[{datetime.now()}] Skipping epoch{epoch_num} - not found")
        continue
    print(f"\n[{datetime.now()}] Evaluating epoch{epoch_num} on holdout...")
    model_ep = YOLO(epoch_weights)
    results_ep = model_ep.val(
        data=DATA_YAML, split="test", save_json=False, save_conf=True,
        plots=False, device="0", name=f"holdout_epoch{epoch_num}_v2",
        project=PROJECT, conf=0.25, iou=0.6,
    )
    n_ep = len(results_ep.box.ap50)
    print(f"  Epoch{epoch_num}: mAP50={results_ep.box.map50:.4f}, mAP50-95={results_ep.box.map:.4f}")
    for i in range(n_ep):
        name = results_ep.names.get(i, f"c{i}")
        print(f"    {name}: AP50={results_ep.box.ap50[i]:.4f}, AP={results_ep.box.ap[i]:.4f}")
    ep_data = {
        "timestamp": datetime.now().isoformat(), "weights": epoch_weights,
        "mAP50": float(results_ep.box.map50), "mAP50-95": float(results_ep.box.map),
        "precision": float(results_ep.box.mp), "recall": float(results_ep.box.mr),
        "per_class": {results_ep.names.get(i,f"c{i}"): {"AP50":float(results_ep.box.ap50[i]),"AP":float(results_ep.box.ap[i])} for i in range(n_ep)}
    }
    with open(os.path.join(PROJECT, f"holdout_epoch{epoch_num}_v2_results.json"), "w") as f:
        json.dump(ep_data, f, indent=2)

print(f"\n[{datetime.now()}] ALL holdout evaluations complete!")
"""

E60_RESUME_SCRIPT = r"""#!/usr/bin/env python3
import os
os.environ["MPLBACKEND"] = "Agg"
from datetime import datetime

print(f"[{datetime.now()}] E60 Resume training starting...")
from ultralytics import YOLO

LAST_PT = "/home/uav/gu/stage4/runs/stage4_p2_fasdd_dfire_tile_e60_from_e20best/weights/last.pt"
DATA_YAML = "/home/uav/gu/projects/FireAndSmoke_3/datasets/stage4_full_tile_sensors3/data.yaml"

print(f"[{datetime.now()}] Loading checkpoint: {LAST_PT}")
model = YOLO(LAST_PT)

print(f"[{datetime.now()}] Starting resume training (plots=False)...")
model.train(
    data=DATA_YAML,
    epochs=60,
    imgsz=960,
    device="0,1",
    batch=16,
    lr0=0.003,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=0,
    optimizer="auto",
    weight_decay=0.0005,
    mosaic=0.15,
    patience=20,
    save_period=5,
    plots=False,
    val=True,
    project="/home/uav/gu/stage4/runs",
    name="stage4_e60_resume_from_last",
    exist_ok=False,
    seed=20260610,
    resume=True,
)

print(f"[{datetime.now()}] E60 Resume training complete!")
"""

PIPELINE_SCRIPT = r"""#!/bin/bash
set -e
LOG_DIR="/home/uav/gu/stage4"

echo "=============================================="
echo "Stage4 Pipeline v2 Started: $(date)"
echo "=============================================="

# Step 1: Holdout eval (quick)
echo "[Step 1/2] Starting Holdout Evaluation..."
echo "[$(date)] Holdout eval starting" >> ${LOG_DIR}/pipeline_status.log
cd /home/uav/gu
MPLBACKEND=Agg python3 ${LOG_DIR}/run_holdout_eval_v2.py 2>&1 | tee -a ${LOG_DIR}/holdout_eval_v2.log
echo "[$(date)] Holdout eval done" >> ${LOG_DIR}/pipeline_status.log
echo "[Step 1/2] Holdout Evaluation COMPLETE!"

# Step 2: Resume E60 training
echo "[Step 2/2] Starting E60 Resume Training..."
echo "[$(date)] E60 resume training starting" >> ${LOG_DIR}/pipeline_status.log
cd /home/uav/gu
MPLBACKEND=Agg python3 ${LOG_DIR}/run_e60_resume_v2.py 2>&1 | tee -a ${LOG_DIR}/stage4_e60_resume_v2.log
echo "[$(date)] E60 resume training done" >> ${LOG_DIR}/pipeline_status.log
echo "[Step 2/2] E60 Resume Training COMPLETE!"

echo "=============================================="
echo "Stage4 Pipeline v2 Finished: $(date)"
echo "=============================================="
"""


def main():
    print("Connecting to server...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    # Kill old processes
    print("Killing old processes...")
    stdin, stdout, stderr = ssh.exec_command(
        'pkill -f run_pipeline 2>/dev/null; pkill -f run_holdout 2>/dev/null; '
        'pkill -f run_e60 2>/dev/null; pkill -f run_e120 2>/dev/null; '
        'pkill -f torch.distributed 2>/dev/null; echo done',
        timeout=30
    )
    stdout.channel.settimeout(30)
    print(stdout.read().decode().strip())

    # Upload scripts
    print("Uploading scripts...")
    sftp = ssh.open_sftp()
    scripts = {
        f'{REMOTE_BASE}/run_holdout_eval_v2.py': HOLDOUT_SCRIPT,
        f'{REMOTE_BASE}/run_e60_resume_v2.py': E60_RESUME_SCRIPT,
        f'{REMOTE_BASE}/run_pipeline_v2.sh': PIPELINE_SCRIPT,
    }
    for path, content in scripts.items():
        with sftp.open(path, 'w') as f:
            f.write(content)
        print(f"  Uploaded: {path}")
    sftp.close()

    # Set permissions
    print("Setting permissions...")
    stdin, stdout, stderr = ssh.exec_command(
        f'chmod +x {REMOTE_BASE}/run_holdout_eval_v2.py {REMOTE_BASE}/run_e60_resume_v2.py {REMOTE_BASE}/run_pipeline_v2.sh',
        timeout=30
    )
    stdout.channel.settimeout(30)
    stdout.read()

    # Start pipeline v2 - use setsid to fully detach from SSH session
    print("Starting pipeline v2 in background...")
    stdin, stdout, stderr = ssh.exec_command(
        'cd /home/uav/gu && setsid bash -c \'MPLBACKEND=Agg nohup bash /home/uav/gu/stage4/run_pipeline_v2.sh '
        '> /home/uav/gu/stage4/pipeline_v2_nohup.log 2>&1 &\' && echo "Pipeline started" && sleep 1',
        timeout=15
    )
    stdout.channel.settimeout(15)
    try:
        print('Start:', stdout.read().decode().strip())
    except Exception:
        print('Start: (command sent, timeout is OK for background process)')

    ssh.close()

    # Wait and check status
    print("Waiting 15 seconds for pipeline to initialize...")
    time.sleep(15)

    print("Checking pipeline status...")
    ssh2 = paramiko.SSHClient()
    ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh2.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    # Check processes
    stdin, stdout, stderr = ssh2.exec_command(
        'ps aux | grep -E "run_pipeline|run_holdout|run_e60|torch.distributed" | grep -v grep',
        timeout=30
    )
    stdout.channel.settimeout(30)
    procs = stdout.read().decode().strip()
    print(f"Processes:\n{procs}")

    # Check log
    stdin, stdout, stderr = ssh2.exec_command(
        'tail -30 /home/uav/gu/stage4/pipeline_v2_nohup.log 2>/dev/null',
        timeout=30
    )
    stdout.channel.settimeout(30)
    log = stdout.read().decode().strip()
    print(f"Log (last 30 lines):\n{log}")

    # Check GPU
    stdin, stdout, stderr = ssh2.exec_command(
        'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv',
        timeout=30
    )
    stdout.channel.settimeout(30)
    gpu = stdout.read().decode().strip()
    print(f"GPU:\n{gpu}")

    ssh2.close()
    print("\n=== Pipeline v2 deployed and running! ===")


if __name__ == '__main__':
    main()
