param(
  [string]$Project = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3",
  [string]$RunRoot = "D:\Researching\Training_After\FireAndSmoke_Stage4",
  [string]$Data = "",
  [string]$Model = "",
  [string]$Pretrained = "",
  [string]$Name = "stage4_scout_p2_full_tile_e20",
  [int]$Epochs = 20,
  [int]$Batch = 8,
  [int]$ImgSize = 960,
  [string]$Device = "0",
  [int]$Workers = 8
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

if (-not $Data) { $Data = Join-Path $Project "datasets\stage4_full_tile_sensors3\data.yaml" }
if (-not $Model) { $Model = Join-Path $Project "models\yolov8m-p2-fire-smoke-3cls.yaml" }
if (-not $Pretrained) { $Pretrained = Join-Path $Project "weights\sensors_yolov8m_3cls_100_best.pt" }
if (-not (Test-Path $Data)) { throw "Missing data yaml: $Data. Run scripts\11_build_highres_tiles.ps1 first, or pass -Data." }
if (-not (Test-Path $Model)) { throw "Missing model yaml: $Model" }
if (-not (Test-Path $Pretrained)) { $Pretrained = "yolov8m.pt" }

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

python (Join-Path $Project "tools\train_yolo_api.py") `
  --model $Model `
  --pretrained $Pretrained `
  --data $Data `
  --epochs $Epochs `
  --batch $Batch `
  --imgsz $ImgSize `
  --device $Device `
  --project (Join-Path $RunRoot "runs") `
  --name $Name `
  --patience 8 `
  --close-mosaic 5 `
  --mosaic 0.20 `
  --mixup 0.0 `
  --copy-paste 0.0 `
  --erasing 0.0 `
  --auto-augment none `
  --save-period 5 `
  --workers $Workers `
  --amp true
