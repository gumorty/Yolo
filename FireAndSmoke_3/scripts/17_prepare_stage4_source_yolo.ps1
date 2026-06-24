param(
  [string]$Project = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3",
  [Parameter(Mandatory=$true)][string]$Root,
  [Parameter(Mandatory=$true)][string]$Out,
  [Parameter(Mandatory=$true)][string]$SourceId,
  [string[]]$Names = @("fire", "other", "smoke"),
  [string[]]$ClassMap = @("0=0", "1=2"),
  [switch]$IncludeEmpty,
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$argsList = @(
  "--root", $Root,
  "--out", $Out,
  "--source-id", $SourceId,
  "--names"
) + $Names + @("--class-map") + $ClassMap
if ($IncludeEmpty) { $argsList += "--include-empty" }
if ($Overwrite) { $argsList += "--overwrite" }

python (Join-Path $Project "tools\prepare_stage4_source_yolo.py") @argsList
