[CmdletBinding()]
param(
    [string] $RunDir = 'D:\zbm\Eigenfluid-SSS\BasisNet\cases\3D_neural_kinematic\results\eigenbasis_k1024_native_exact_vorticity_rayleigh_full',
    [string] $Executable = 'D:\zbm\Eigenfluid-SSS\BasisNet\cases\3D_neural_kinematic\native\build\eigenfluid_native.exe',
    [int] $StaleMinutes = 45
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$pidPath = Join-Path $RunDir 'trainer.pid'
$historyPath = Join-Path $RunDir 'history.jsonl'
$checkpointPath = Join-Path $RunDir 'latest.bin'
$summaryPath = Join-Path $RunDir 'summary.json'
$statePath = Join-Path $RunDir 'monitor_state.json'
$failurePath = Join-Path $RunDir 'failure.txt'

function Start-NativeTraining {
    $info = [System.Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.Arguments = '--train --device 0 --seed 250415657 --run-dir "' + $RunDir + '"'
    $info.WorkingDirectory = 'D:\zbm\Eigenfluid-SSS'
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $info
    [void] $process.Start()
    Set-Content -LiteralPath $pidPath -Value $process.Id
    return $process.Id
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$summary = $null
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
}
if ($null -ne $summary -and $summary.status -eq 'complete') {
    $state = [ordered]@{
        checked_at = (Get-Date).ToString('o')
        status = 'complete'
        step = 10000
        restarted = $false
    }
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath
    $state | ConvertTo-Json -Compress
    exit 0
}

$trainingProcess = $null
if (Test-Path -LiteralPath $pidPath -PathType Leaf) {
    $trainingPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $trainingProcess = Get-Process -Id $trainingPid -ErrorAction SilentlyContinue
}
$lastRecord = $null
if (Test-Path -LiteralPath $historyPath -PathType Leaf) {
    $lastLine = Get-Content -LiteralPath $historyPath -Tail 1
    if ($lastLine) {
        $lastRecord = $lastLine | ConvertFrom-Json
    }
}
$stale = $false
if ($null -ne $trainingProcess -and (Test-Path -LiteralPath $historyPath)) {
    $age = (Get-Date) - (Get-Item -LiteralPath $historyPath).LastWriteTime
    $stale = $age.TotalMinutes -ge $StaleMinutes
}
$restarted = $false
if ($stale -and (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
    Stop-Process -Id $trainingProcess.Id -Force
    Start-Sleep -Seconds 2
    $trainingProcess = $null
}
if ($null -eq $trainingProcess) {
    if (Test-Path -LiteralPath $failurePath -PathType Leaf) {
        $failure = (Get-Content -LiteralPath $failurePath -Raw).Trim()
    } else {
        $failure = $null
    }
    $newPid = Start-NativeTraining
    $trainingProcess = Get-Process -Id $newPid
    $restarted = $true
}

$gpu = (& nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader).Trim()
$state = [ordered]@{
    checked_at = (Get-Date).ToString('o')
    status = 'running'
    pid = $trainingProcess.Id
    step = if ($null -ne $lastRecord) { $lastRecord.step } else { 0 }
    loss_sum = if ($null -ne $lastRecord) { $lastRecord.loss_sum } else { $null }
    history_stale = $stale
    restarted = $restarted
    latest_checkpoint = (Test-Path -LiteralPath $checkpointPath -PathType Leaf)
    gpu = $gpu
}
$state | ConvertTo-Json | Set-Content -LiteralPath $statePath
$state | ConvertTo-Json -Compress
