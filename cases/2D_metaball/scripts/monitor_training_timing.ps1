param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid,
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [Parameter(Mandatory = $true)]
    [string]$ConfigPath,
    [Parameter(Mandatory = $true)]
    [string]$InitialCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$ValidationPack,
    [Parameter(Mandatory = $true)]
    [string]$Worktree
)

$ErrorActionPreference = "Stop"
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$dataDir = Join-Path $RunDir "data"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

$metricsCsv = Join-Path $dataDir "training_metrics_timing.csv"
$metricsJson = Join-Path $dataDir "training_metrics_timing.json"
$checkpointJson = Join-Path $dataDir "checkpoint_timing.json"
$manifestJson = Join-Path $dataDir "training_run_manifest.json"
$statusJson = Join-Path $dataDir "training_timing_status.json"
$summaryJson = Join-Path $dataDir "training_timing_summary.json"
$trainLog = Join-Path $RunDir "hardcoded_train.log"

function Write-JsonAtomic {
    param([object]$Value, [string]$Path, [int]$Depth = 12)
    $temporary = "$Path.tmp"
    $json = ConvertTo-Json -InputObject $Value -Depth $Depth
    if ($null -eq $json) {
        $json = "null"
    }
    Set-Content -LiteralPath $temporary -Value $json -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-Sha256WithRetry {
    param([string]$Path)
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try {
            $result = Get-FileHash -LiteralPath $Path -Algorithm SHA256
            if ($null -ne $result -and -not [string]::IsNullOrWhiteSpace($result.Hash)) {
                return $result.Hash.ToLowerInvariant()
            }
        } catch {
            if ($attempt -eq 10) {
                throw
            }
        }
        Start-Sleep -Milliseconds 200
    }
    throw "could not hash checkpoint after retries: $Path"
}

function File-Record {
    param([string]$Path)
    $item = Get-Item -LiteralPath $Path
    return [ordered]@{
        path = $item.FullName
        bytes = $item.Length
        sha256 = Get-Sha256WithRetry $Path
        last_write_time_local = $item.LastWriteTime.ToString("o")
        last_write_time_utc = $item.LastWriteTimeUtc.ToString("o")
    }
}

function Parse-Metrics {
    param([datetime]$ProcessStartLocal)
    if (-not (Test-Path -LiteralPath $trainLog)) {
        return @()
    }
    $pattern = '^\[(?<timestamp>[^\]]+)\]\s+epoch=(?<epoch>\d+)\s+beta=(?<beta>[-+0-9.eE]+)\s+alpha=(?<alpha>[-+0-9.eE]+)\s+train_psi_R=(?<train>[-+0-9.eE]+)\s+validation_psi_R=(?<validation>[-+0-9.eE]+)\s+val_p95=(?<p95>[-+0-9.eE]+)\s+lr=(?<lr>[-+0-9.eE]+)\s+orth=(?<orth>[-+0-9.eE]+)\s+peeled_rank=(?<rank>\d+)$'
    $records = @()
    $previousTime = $null
    foreach ($line in Get-Content -LiteralPath $trainLog) {
        if ($line -notmatch $pattern) {
            continue
        }
        $eventTime = [datetime]::ParseExact(
            $Matches.timestamp, "yyyy-MM-dd HH:mm:ss", $culture,
            [System.Globalization.DateTimeStyles]::None
        )
        $interval = if ($null -eq $previousTime) {
            ($eventTime - $ProcessStartLocal).TotalSeconds
        } else {
            ($eventTime - $previousTime).TotalSeconds
        }
        $records += [pscustomobject][ordered]@{
            epoch = [int]$Matches.epoch
            timestamp_local = $eventTime.ToString("yyyy-MM-ddTHH:mm:ss")
            elapsed_from_process_start_seconds = [math]::Round(
                ($eventTime - $ProcessStartLocal).TotalSeconds, 3
            )
            interval_seconds = [math]::Round($interval, 3)
            beta = [double]::Parse($Matches.beta, $culture)
            alpha = [double]::Parse($Matches.alpha, $culture)
            train_rayleigh = [double]::Parse($Matches.train, $culture)
            validation_rayleigh = [double]::Parse($Matches.validation, $culture)
            validation_p95 = [double]::Parse($Matches.p95, $culture)
            learning_rate = [double]::Parse($Matches.lr, $culture)
            orthogonality_error = [double]::Parse($Matches.orth, $culture)
            peeled_rank = [int]$Matches.rank
        }
        $previousTime = $eventTime
    }
    return @($records)
}

function Checkpoint-Records {
    param([datetime]$ProcessStartLocal)
    $checkpointDir = Join-Path $RunDir "checkpoints"
    if (-not (Test-Path -LiteralPath $checkpointDir)) {
        return @()
    }
    $records = @()
    $checkpointFiles = Get-ChildItem -LiteralPath $checkpointDir -File |
        Where-Object { $_.Extension -in @(".npz", ".json") } |
        Sort-Object Name
    foreach ($item in $checkpointFiles) {
        $epoch = $null
        if ($item.Name -match 'epoch(?<epoch>\d+)') {
            $epoch = [int]$Matches.epoch
        }
        $records += [pscustomobject][ordered]@{
            name = $item.Name
            epoch = $epoch
            timestamp_local = $item.LastWriteTime.ToString("o")
            elapsed_from_process_start_seconds = [math]::Round(
                ($item.LastWriteTime - $ProcessStartLocal).TotalSeconds, 3
            )
            bytes = $item.Length
            sha256 = Get-Sha256WithRetry $item.FullName
        }
    }
    return @($records)
}

$process = [System.Diagnostics.Process]::GetProcessById($TrainingPid)
$processStartLocal = $process.StartTime
$processStartUtc = $process.StartTime.ToUniversalTime()
$monitorStartLocal = Get-Date
$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$safeDirectoryArgument = "safe.directory=$Worktree"
$gitCommit = (& git -c $safeDirectoryArgument -C $Worktree rev-parse HEAD).Trim()
$gitBranch = (& git -c $safeDirectoryArgument -C $Worktree branch --show-current).Trim()
$gitDirty = (@(& git -c $safeDirectoryArgument -C $Worktree status --porcelain)).Count -gt 0

$manifest = [ordered]@{
    schema_version = 1
    timing = [ordered]@{
        timer = "System.Diagnostics.Process StartTime/ExitTime"
        process_id = $TrainingPid
        process_start_time_local = $processStartLocal.ToString("o")
        process_start_time_utc = $processStartUtc.ToString("o")
        monitor_start_time_local = $monitorStartLocal.ToString("o")
        metric_event_resolution = "one second (trainer log timestamps)"
        total_runtime_resolution = "native Windows process timestamp resolution"
        polling_interval_seconds = 5
    }
    source = [ordered]@{
        worktree = $Worktree
        git_commit = $gitCommit
        git_branch = $gitBranch
        git_worktree_dirty_at_monitor_start = $gitDirty
    }
    artifacts = [ordered]@{
        config = File-Record $ConfigPath
        initial_checkpoint = File-Record $InitialCheckpoint
        validation_pack = File-Record $ValidationPack
    }
    configuration = $config
    recorded_metrics = @(
        "train_rayleigh", "validation_rayleigh", "validation_p95",
        "learning_rate", "orthogonality_error", "peeled_rank",
        "beta", "alpha", "elapsed_time"
    )
}
Write-JsonAtomic -Value $manifest -Path $manifestJson -Depth 20

$lastMetricCount = -1
$lastCheckpointSignature = ""
while (-not $process.HasExited) {
    $metrics = @(Parse-Metrics $processStartLocal)
    if ($metrics.Count -ne $lastMetricCount) {
        $metrics | Export-Csv -LiteralPath $metricsCsv -NoTypeInformation -Encoding utf8
        Write-JsonAtomic -Value $metrics -Path $metricsJson
        $lastMetricCount = $metrics.Count
    }

    $checkpoints = @(Checkpoint-Records $processStartLocal)
    $checkpointSignature = ($checkpoints | ForEach-Object {
        "$($_.name):$($_.bytes):$($_.timestamp_local)"
    }) -join '|'
    if ($checkpointSignature -ne $lastCheckpointSignature) {
        Write-JsonAtomic -Value $checkpoints -Path $checkpointJson
        $lastCheckpointSignature = $checkpointSignature
    }

    $now = Get-Date
    $lastMetric = if ($metrics.Count -gt 0) { $metrics[-1] } else { $null }
    $status = [ordered]@{
        status = "running"
        process_id = $TrainingPid
        process_start_time_local = $processStartLocal.ToString("o")
        sampled_time_local = $now.ToString("o")
        elapsed_wall_seconds = [math]::Round(($now - $processStartLocal).TotalSeconds, 3)
        validation_records = $metrics.Count
        last_metric = $lastMetric
        checkpoint_records = $checkpoints.Count
    }
    Write-JsonAtomic -Value $status -Path $statusJson
    Start-Sleep -Seconds 5
    $process.Refresh()
}

$exitTimeLocal = $process.ExitTime
$finalMetrics = @(Parse-Metrics $processStartLocal)
$finalMetrics | Export-Csv -LiteralPath $metricsCsv -NoTypeInformation -Encoding utf8
Write-JsonAtomic -Value $finalMetrics -Path $metricsJson
$finalCheckpoints = @(Checkpoint-Records $processStartLocal)
Write-JsonAtomic -Value $finalCheckpoints -Path $checkpointJson

$summary = [ordered]@{
    status = "process_exited"
    process_id = $TrainingPid
    exit_code = $process.ExitCode
    process_start_time_local = $processStartLocal.ToString("o")
    process_exit_time_local = $exitTimeLocal.ToString("o")
    exact_wall_seconds = [math]::Round(
        ($exitTimeLocal - $processStartLocal).TotalSeconds, 6
    )
    exact_wall_minutes = [math]::Round(
        ($exitTimeLocal - $processStartLocal).TotalMinutes, 6
    )
    validation_records = $finalMetrics.Count
    last_metric = if ($finalMetrics.Count -gt 0) { $finalMetrics[-1] } else { $null }
    checkpoint_records = $finalCheckpoints.Count
}
Write-JsonAtomic -Value $summary -Path $summaryJson
Write-JsonAtomic -Value $summary -Path $statusJson
