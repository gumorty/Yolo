param(
  [string]$Project = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3",
  [string]$Config = "",
  [string]$Out = "",
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
if (-not $Config) { $Config = Join-Path $Project "configs\stage4_dataset_mix_template.yaml" }
if (-not $Out) { $Out = Join-Path $Project "datasets\stage4_mixed_3cls" }

$argsList = @("--config", $Config, "--out", $Out)
if ($Overwrite) { $argsList += "--overwrite" }

python (Join-Path $Project "tools\build_stage4_mixed_dataset.py") @argsList
