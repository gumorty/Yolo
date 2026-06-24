param(
  [string]$Root = "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3",
  [string]$SelfDataset = "C:\Users\jsj506\Desktop\YOLO\Yolo\fire_pic\fire_dataset\fire_dataset_all"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path $PSScriptRoot -Parent

$argsList = @("--root", $Root)
if (Test-Path $SelfDataset) {
  $argsList += @("--self2", $SelfDataset)
}

python (Join-Path $Project "tools\build_strict_datasets.py") @argsList
