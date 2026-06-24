param(
  [string]$Project = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3",
  [string]$Data = "",
  [string]$Out = ""
)

$ErrorActionPreference = "Stop"
if (-not $Data) { $Data = Join-Path $Project "datasets\stage4_full_tile_sensors3\data.yaml" }
if (-not $Out) { $Out = Join-Path $Project "reports\stage4_full_tile_sensors3.audit.json" }
if (-not (Test-Path $Data)) { throw "Missing data yaml: $Data" }

python (Join-Path $Project "tools\dataset_audit.py") --data $Data --out $Out
Write-Host "Audit written to $Out"
