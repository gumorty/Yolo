#!/usr/bin/env bash
set -euo pipefail

echo "== User / host =="
whoami
hostname
pwd

echo
echo "== Login users =="
who || true
w || true

echo
echo "== GPU =="
nvidia-smi || true

echo
echo "== Python =="
python3 --version || true
python --version || true

echo
echo "== PyTorch / Ultralytics =="
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as exc:
    print("torch check failed:", exc)

try:
    import ultralytics
    print("ultralytics:", ultralytics.__version__)
except Exception as exc:
    print("ultralytics check failed:", exc)
PY
