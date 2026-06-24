$ErrorActionPreference = "Stop"

$root = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3"
$runName = "stage6_d1_local3070_corrected_e20_b2_i960_adamw_lr1e4"
$weights = Join-Path $root "runs_stage6_local3070\$runName\weights\best.pt"

if (-not (Test-Path -LiteralPath $weights)) {
    throw "Missing best.pt. Training is not finished yet: $weights"
}

Set-Location $root

python tools\stage6_threshold_scan.py `
  --model $weights `
  --data "$root\datasets\stage4_eval\public_hard_holdout\data.yaml" `
  --split val `
  --imgsz 960 `
  --iou 0.50 `
  --out "$root\reports\stage6_threshold_scan_d1_local3070_corrected_public_hard.csv"

python tools\stage6_threshold_scan.py `
  --model $weights `
  --data "$root\datasets\stage6_eval\uav_false_alarm_v2\data.yaml" `
  --split val `
  --imgsz 960 `
  --iou 0.50 `
  --out "$root\reports\stage6_threshold_scan_d1_local3070_corrected_false_alarm_v2.csv"

Write-Host "Stage6 D1 corrected local3070 evaluation complete."
