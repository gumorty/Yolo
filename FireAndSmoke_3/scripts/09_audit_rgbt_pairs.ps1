param(
  [string]$RgbDir,
  [string]$ThermalDir,
  [string]$Root = "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3"
)

$ErrorActionPreference = "Stop"
if (-not $RgbDir -or -not $ThermalDir) {
  throw "Usage: .\scripts\09_audit_rgbt_pairs.ps1 -RgbDir <rgb_images> -ThermalDir <thermal_images>"
}

$Project = Split-Path $PSScriptRoot -Parent
$Reports = Join-Path $Root "reports"
New-Item -ItemType Directory -Force $Reports | Out-Null
python (Join-Path $Project "tools\rgbt_pair_audit.py") `
  --rgb-dir $RgbDir `
  --thermal-dir $ThermalDir `
  --out (Join-Path $Reports "rgbt_pair_audit.json")
