param(
  [string]$Project = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3",
  [string]$Data = "",
  [string]$Out = "",
  [int]$TileSize = 960,
  [double]$Overlap = 0.25,
  [double]$MinVisibility = 0.35,
  [double]$NegativeRatio = 0.10,
  [switch]$CopyFull,
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
if (-not $Data) { $Data = Join-Path $Project "datasets\strict_sensors_3cls\data.yaml" }
if (-not $Out) { $Out = Join-Path $Project "datasets\stage4_full_tile_sensors3" }

$argsList = @(
  "--data", $Data,
  "--out", $Out,
  "--tile-size", $TileSize,
  "--overlap", $Overlap,
  "--min-visibility", $MinVisibility,
  "--negative-ratio", $NegativeRatio,
  "--positive-classes", "fire", "smoke"
)
if ($CopyFull) { $argsList += "--copy-full" }
if ($Overwrite) { $argsList += "--overwrite" }

python (Join-Path $Project "tools\build_highres_tile_dataset.py") @argsList
