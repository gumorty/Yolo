param(
  [string]$Root = "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3",
  [int]$Epochs = 120,
  [int]$Batch = 8,
  [int]$ImgSize = 960,
  [string]$Device = "0",
  [string]$Pretrained = ""
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Project = Split-Path $PSScriptRoot -Parent
$Data = Join-Path $Root "datasets\strict_sensors_2cls\data.yaml"
$Model = Join-Path $Project "models\yolov8m-p2-fire-smoke-2cls.yaml"
$Runs = Join-Path $Root "runs"

if (-not (Test-Path $Data)) { throw "Missing dataset yaml: $Data. Run scripts\02_build_strict_datasets.ps1 first." }
if (-not $Pretrained) {
  $candidate = Join-Path $Root "initial_weights\sensors_yolov8m_2cls_100_best.pt"
  if (Test-Path $candidate) { $Pretrained = $candidate } else { $Pretrained = "yolov8m.pt" }
}

Write-Host "== Train YOLOv8m-P2 Sensors 2-class =="
Write-Host "Data: $Data"
Write-Host "Pretrained: $Pretrained"

yolo task=detect mode=train `
  model=$Model `
  pretrained=$Pretrained `
  data=$Data `
  epochs=$Epochs `
  batch=$Batch `
  imgsz=$ImgSize `
  device=$Device `
  project=$Runs `
  name="stage3_yolov8m_p2_sensors2_e$Epochs" `
  patience=35 `
  close_mosaic=20 `
  mosaic=0.7 `
  mixup=0.05 `
  copy_paste=0.05 `
  workers=8
