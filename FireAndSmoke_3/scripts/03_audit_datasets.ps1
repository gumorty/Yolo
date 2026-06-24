param(
  [string]$Root = "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path $PSScriptRoot -Parent
$Reports = Join-Path $Root "reports"
New-Item -ItemType Directory -Force $Reports | Out-Null

$yamls = @(
  Join-Path $Root "datasets\strict_sensors_3cls\data.yaml",
  Join-Path $Root "datasets\strict_sensors_2cls\data.yaml",
  Join-Path $Root "datasets\self_fire_pic_2cls_standalone\data.yaml"
)

foreach ($yaml in $yamls) {
  if (Test-Path $yaml) {
    $name = Split-Path (Split-Path $yaml -Parent) -Leaf
    python (Join-Path $Project "tools\dataset_audit.py") --data $yaml --out (Join-Path $Reports "$name.audit.json")
  } else {
    Write-Host "Skip missing dataset yaml: $yaml"
  }
}
