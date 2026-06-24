#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-$HOME/gu}"
ENV_DIR="$BASE/envs/fire_smoke_yolo"

mkdir -p "$BASE/envs" "$BASE/logs"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# CUDA 12.4 driver can run cu121 wheels. This avoids sudo and system CUDA edits.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.3.0 opencv-python-headless pyyaml pandas tqdm matplotlib seaborn

python - <<'PY'
import torch, ultralytics, cv2, yaml, pandas
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("ultralytics:", ultralytics.__version__)
print("basic packages ok")
PY

echo "Environment ready: $ENV_DIR"
