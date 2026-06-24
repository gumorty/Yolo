param(
  [string]$Project = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3",
  [string]$Out = ""
)

$ErrorActionPreference = "Stop"
if (-not $Out) { $Out = Join-Path $Project "reports\stage4_preflight_check.json" }

python (Join-Path $Project "tools\stage4_preflight_check.py") --project $Project --out $Out
