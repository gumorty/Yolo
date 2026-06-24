param()

$ErrorActionPreference = "Stop"

Write-Host "== Python =="
python --version

Write-Host "`n== PyTorch / CUDA =="
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host "`n== Ultralytics =="
python -c "import ultralytics; print(ultralytics.__version__)"

Write-Host "`n== Required packages =="
python -c "import cv2, yaml; print('opencv/yaml ok')"

Write-Host "`nEnvironment check complete."
