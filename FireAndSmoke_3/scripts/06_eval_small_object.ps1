param(
  [string]$Root = "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3",
  [string]$Model = "",
  [string]$Data = "",
  [int]$ImgSize = 960,
  [double]$Conf = 0.20,
  [double]$Iou = 0.50,
  [int]$MaxImages = 0
)

$ErrorActionPreference = "Stop"
$Project = Split-Path $PSScriptRoot -Parent
$Reports = Join-Path $Root "reports"
New-Item -ItemType Directory -Force $Reports | Out-Null

if (-not $Model) {
  $Model = Join-Path $Root "runs\stage3_yolov8m_p2_sensors3_e120\weights\best.pt"
}
if (-not $Data) {
  $Data = Join-Path $Root "datasets\strict_sensors_3cls\data.yaml"
}
if (-not (Test-Path $Model)) { throw "Missing model: $Model" }
if (-not (Test-Path $Data)) { throw "Missing dataset yaml: $Data" }

$out = Join-Path $Reports ("small_object_eval_" + (Split-Path (Split-Path $Model -Parent) -Parent | Split-Path -Leaf) + ".json")
$argsList = @("--model", $Model, "--data", $Data, "--imgsz", $ImgSize, "--conf", $Conf, "--iou", $Iou, "--out", $out)
if ($MaxImages -gt 0) { $argsList += @("--max-images", $MaxImages) }
python (Join-Path $Project "tools\evaluate_small_objects.py") @argsList
