param(
  [string]$Project = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3",
  [string]$Data = "",
  [string]$Out = "",
  [int]$MaxImages = 500,
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
if (-not $Data) { $Data = Join-Path $Project "datasets\stage4_mixed_3cls\data.yaml" }
if (-not $Out) { $Out = Join-Path $Project "datasets\stage4_eval\public_hard_holdout" }

$argsList = @(
  "--data", $Data,
  "--out", $Out,
  "--max-images", $MaxImages,
  "--target-classes", "fire", "smoke"
)
if ($Overwrite) { $argsList += "--overwrite" }

python (Join-Path $Project "tools\build_yolo_holdout.py") @argsList
