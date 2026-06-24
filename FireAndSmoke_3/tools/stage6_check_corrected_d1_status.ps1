$ErrorActionPreference = "Stop"

$root = "D:\Researching\Yolo\FireAndSmoke\FireAndSmoke_3"
$run = Join-Path $root "runs_stage6_local3070\stage6_d1_local3070_corrected_e20_b2_i960_adamw_lr1e4"
$results = Join-Path $run "results.csv"
$log = Join-Path $root "logs\stage6_d1_local3070_corrected_e20_b2_i960_adamw_lr1e4.out.log"

Write-Host "=== corrected D1 进程 ==="
Get-Process -Id 26572 -ErrorAction SilentlyContinue | Select-Object Id, CPU, StartTime, Responding, Path | Format-List

Write-Host "=== results.csv ==="
if (Test-Path -LiteralPath $results) {
    Import-Csv $results | Format-Table -AutoSize
} else {
    Write-Host "尚未生成 results.csv"
}

Write-Host "=== 最近日志 ==="
if (Test-Path -LiteralPath $log) {
    Get-Content -LiteralPath $log -Tail 40
} else {
    Write-Host "尚未生成日志"
}

Write-Host "=== GPU 状态 ==="
nvidia-smi
