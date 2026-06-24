param(
  [string]$Project = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3",
  [string]$ImageDir = "",
  [string]$OldModel = "",
  [Parameter(Mandatory=$true)][string]$NewModel,
  [string]$Out = "",
  [int]$MaxImages = 500,
  [switch]$IncludeSliced
)

$ErrorActionPreference = "Stop"
if (-not $ImageDir) { $ImageDir = Join-Path $Project "datasets\stage4_eval\public_hard_holdout\images" }
if (-not $OldModel) { $OldModel = Join-Path $Project "weights\sensors_yolov8m_3cls_100_best.pt" }
if (-not $Out) { $Out = Join-Path $Project "reports\stage4_public_holdout_compare.json" }
if (-not (Test-Path $ImageDir)) { throw "Missing holdout image dir: $ImageDir" }
if (-not (Test-Path $OldModel)) { throw "Missing old model: $OldModel" }
if (-not (Test-Path $NewModel)) { throw "Missing new model: $NewModel" }

$argsList = @(
  "--image-dir", $ImageDir,
  "--old-model", $OldModel,
  "--new-model", $NewModel,
  "--max-images", $MaxImages,
  "--seed", 20260610,
  "--require-small-fire-smoke",
  "--out", $Out
)
if ($IncludeSliced) { $argsList += "--include-sliced" }

python (Join-Path $Project "tools\compare_detection_configs.py") @argsList
