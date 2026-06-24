param(
  [string]$Root = "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3",
  [string]$FirstRuns = "C:\Users\jsj506\Desktop\FireAndSmoke\FireSmoke_Reproduction\repos\runs",
  [string]$SecondRuns = "C:\Users\jsj506\Desktop\FireAndSmoke\FireSmoke_Reproduction_Two\repos\runs"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path $PSScriptRoot -Parent
$Reports = Join-Path $Root "reports"
New-Item -ItemType Directory -Force $Reports | Out-Null

python (Join-Path $Project "tools\collect_previous_results.py") `
  --roots $FirstRuns $SecondRuns `
  --out (Join-Path $Reports "previous_training_summary.csv")
