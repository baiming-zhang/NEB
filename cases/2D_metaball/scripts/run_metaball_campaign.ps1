param(
    [string]$ResultRoot = "",
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
$caseRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$workspace = Resolve-Path (Join-Path $caseRoot "..\..\..")
$python = "C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
$env:PYTHONPATH = "C:\Windows\System32\autoslides\Lib\site-packages"
$config = Join-Path $caseRoot "configs\k1024_metaball_siren_raw_fp32_5000.json"
if (-not $ResultRoot) {
    $ResultRoot = Join-Path $caseRoot "results\full_fp32_k1024_5000_siren_omega32"
}
$ResultRoot = [System.IO.Path]::GetFullPath($ResultRoot)
$inputDir = Join-Path $ResultRoot "inputs"
$runDir = Join-Path $ResultRoot "run_001"
$checkpointDir = Join-Path $runDir "checkpoints"
$statePath = Join-Path $ResultRoot "campaign_state.json"
New-Item -ItemType Directory -Force -Path $inputDir, $runDir, $checkpointDir | Out-Null

function Write-State([string]$status, [int]$epoch, [string]$message) {
    $state = [ordered]@{
        status = $status
        current_epoch = $epoch
        target_epochs = 5000
        modes = 1024
        method = "native CUDA Jet3 + spatial-SIREN + raw"
        precision = "all float32"
        updated_at = (Get-Date).ToString("o")
        message = $message
        run_dir = $runDir
    }
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

try {
    $initial = Join-Path $inputDir "initial_epoch0000.npz"
    $validation = Join-Path $inputDir "fixed_validation_scipy_sobol_v1.npz"
    if (-not (Test-Path -LiteralPath $initial)) {
        Write-State "preparing" 0 "creating deterministic K=1024 inputs"
        & $python (Join-Path $caseRoot "scripts\prepare_metaball_run.py") `
            --config $config --output-dir $inputDir
        if ($LASTEXITCODE -ne 0) { throw "input preparation failed with $LASTEXITCODE" }
    }

    if (-not $SkipPreflight) {
        $preflightMarker = Join-Path $ResultRoot "preflight_complete.json"
        if (-not (Test-Path -LiteralPath $preflightMarker)) {
            $preflightDir = Join-Path $ResultRoot "preflight_k1024"
            Write-State "preflight" 0 "one K=1024 epoch with 4096 points"
            & $python (Join-Path $caseRoot "scripts\train_metaball.py") `
                --resume $initial --run-dir $preflightDir `
                --validation-pack $validation --self-test-epochs 1 --self-test-points 4096
            if ($LASTEXITCODE -ne 0) { throw "K=1024 preflight failed with $LASTEXITCODE" }
            $preflightLine = Get-Content -LiteralPath (Join-Path $preflightDir "hardcoded_train.log") |
                Select-String -Pattern "self-test epoch=.*rank=" | Select-Object -Last 1
            if (-not $preflightLine -or $preflightLine.Line -notmatch "rank=(\d+)") {
                throw "K=1024 preflight did not report an effective raw rank"
            }
            $preflightRank = [int]$Matches[1]
            if ($preflightRank -lt 512) {
                throw "K=1024 preflight rank gate failed: $preflightRank < 512"
            }
            [ordered]@{
                complete = $true
                modes = 1024
                points = 4096
                effective_raw_rank = $preflightRank
                minimum_acceptable_rank = 512
                completed_at = (Get-Date).ToString("o")
            } | ConvertTo-Json | Set-Content -LiteralPath $preflightMarker -Encoding UTF8
        }
    }

    $completion = Join-Path $runDir "completion.json"
    if (Test-Path -LiteralPath $completion) {
        $done = Get-Content -Raw -LiteralPath $completion | ConvertFrom-Json
        if ($done.complete -and [int]$done.final_epoch -ge 5000) {
            Write-State "complete" ([int]$done.final_epoch) "training already complete"
            exit 0
        }
    }

    $latest = Join-Path $checkpointDir "hardcoded_latest.npz"
    $resume = if (Test-Path -LiteralPath $latest) { $latest } else { $initial }
    $resumeEpoch = 0
    $resumeSidecar = [System.IO.Path]::ChangeExtension($resume, ".json")
    if (Test-Path -LiteralPath $resumeSidecar) {
        $resumeEpoch = [int]((Get-Content -Raw -LiteralPath $resumeSidecar | ConvertFrom-Json).epoch)
    }
    Write-State "training" $resumeEpoch "training or resuming from latest valid checkpoint"
    & $python (Join-Path $caseRoot "scripts\train_metaball.py") `
        --resume $resume --run-dir $runDir --validation-pack $validation
    if ($LASTEXITCODE -ne 0) { throw "training failed with $LASTEXITCODE" }

    $done = Get-Content -Raw -LiteralPath $completion | ConvertFrom-Json
    Write-State "evaluating" ([int]$done.final_epoch) "rendering geometry and all 1024 modes"
    & $python (Join-Path $caseRoot "scripts\evaluate_metaball.py") --run-dir $runDir
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed with $LASTEXITCODE" }
    Write-State "complete" ([int]$done.final_epoch) "training and evaluation complete"
}
catch {
    Write-State "failed" 0 $_.Exception.Message
    throw
}
