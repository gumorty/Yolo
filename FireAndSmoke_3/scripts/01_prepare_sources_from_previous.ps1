param(
  [string]$Root = "C:\Users\jsj506\Desktop\FireAndSmoke\FireAndSmoke_3",
  [string]$FirstRoot = "C:\Users\jsj506\Desktop\FireAndSmoke\FireSmoke_Reproduction\repos",
  [string]$SecondRoot = "C:\Users\jsj506\Desktop\FireAndSmoke\FireSmoke_Reproduction_Two\repos"
)

$ErrorActionPreference = "Stop"

$Datasets = Join-Path $Root "datasets"
$Weights = Join-Path $Root "initial_weights"
New-Item -ItemType Directory -Force $Datasets, $Weights | Out-Null

function Copy-DirIfExists($src, $dst) {
  if (Test-Path $src) {
    Write-Host "Copy $src -> $dst"
    if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
    Copy-Item $src $dst -Recurse -Force
  } else {
    Write-Host "Missing optional source: $src"
  }
}

Copy-DirIfExists `
  (Join-Path $FirstRoot "datasets\raw_yolov8") `
  (Join-Path $Datasets "raw_yolov8")

Copy-DirIfExists `
  (Join-Path $FirstRoot "datasets\binary_fire_smoke") `
  (Join-Path $Datasets "binary_fire_smoke")

$weightMap = @{
  "sensors_yolov8m_2cls_100_best.pt" = Join-Path $FirstRoot "runs\yolov8m_firesmoke_2cls_100\weights\best.pt"
  "sensors_yolov8m_3cls_100_best.pt" = Join-Path $FirstRoot "runs\yolov8m_firesmoke_3cls_100\weights\best.pt"
  "sensors_yolov10m_3cls_100_best.pt" = Join-Path $FirstRoot "runs\yolov10m_firesmoke_3cls_100\weights\best.pt"
  "augmented_yolov8m_2cls_150_best.pt" = Join-Path $SecondRoot "runs\yolov8m_augmented_2cls_e1505\weights\best.pt"
  "augmented_yolov8m_3cls_150_best.pt" = Join-Path $SecondRoot "runs\yolov8m_augmented_3cls_e150\weights\best.pt"
}

foreach ($name in $weightMap.Keys) {
  $src = $weightMap[$name]
  if (Test-Path $src) {
    Copy-Item $src (Join-Path $Weights $name) -Force
    Write-Host "Copied weight: $name"
  } else {
    Write-Host "Missing optional weight: $src"
  }
}

Write-Host "Source preparation complete: $Root"
