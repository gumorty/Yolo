param(
  [string]$Root = "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3",
  [int]$Epochs = 120,
  [int]$Batch = 8,
  [int]$ImgSize = 960,
  [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path $PSScriptRoot -Parent
$Reports = Join-Path $Root "reports"
New-Item -ItemType Directory -Force $Reports | Out-Null
$Status = Join-Path $Reports "stage3_overnight_status.log"
$OutLog = Join-Path $Reports "stage3_overnight_train.log"
$ErrLog = Join-Path $Reports "stage3_overnight_train.err.log"

"Stage3 training started at $(Get-Date)" | Tee-Object -FilePath $Status

& (Join-Path $Project "scripts\04_train_stage3_p2_sensors3.ps1") `
  -Root $Root -Epochs $Epochs -Batch $Batch -ImgSize $ImgSize -Device $Device `
  *> $OutLog
"DONE: stage3_yolov8m_p2_sensors3_e$Epochs" | Tee-Object -FilePath $Status -Append

& (Join-Path $Project "scripts\05_train_stage3_p2_sensors2.ps1") `
  -Root $Root -Epochs $Epochs -Batch $Batch -ImgSize $ImgSize -Device $Device `
  *>> $OutLog
"DONE: stage3_yolov8m_p2_sensors2_e$Epochs" | Tee-Object -FilePath $Status -Append

"Stage3 training finished at $(Get-Date)" | Tee-Object -FilePath $Status -Append
Write-Host "Stdout/stderr log: $OutLog"
Write-Host "Status log: $Status"
