param(
  [string]$Project = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3",
  [string]$Data = "",
  [string]$Out = ""
)

$ErrorActionPreference = "Stop"
if (-not $Data) { $Data = Join-Path $Project "datasets\stage4_full_tile_sensors3\data.yaml" }
if (-not $Out) { $Out = Join-Path $Project "reports\stage4_readiness.json" }
if (-not (Test-Path $Data)) { throw "Missing data yaml: $Data" }

python (Join-Path $Project "tools\check_stage4_readiness.py") --data $Data --out $Out
