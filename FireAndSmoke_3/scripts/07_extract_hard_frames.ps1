param(
  [string]$Source = "D:\Researching\Yolo\Yolo\docs\video",
  [string]$Out = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3\artifacts\hard_frames",
  [int]$Every = 30,
  [int]$MaxFramesPerVideo = 300
)

$ErrorActionPreference = "Stop"
$Project = Split-Path $PSScriptRoot -Parent

$argsList = @("--source", $Source, "--out", $Out, "--every", $Every)
if ($MaxFramesPerVideo -gt 0) { $argsList += @("--max-frames-per-video", $MaxFramesPerVideo) }
python (Join-Path $Project "tools\extract_video_frames_for_hard_mining.py") @argsList
