param([switch]$SkipPreflight)

$ErrorActionPreference = "Stop"
$caseRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = "C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
$env:PYTHONPATH = "C:\Windows\System32\autoslides\Lib\site-packages"
$config = Join-Path $caseRoot "configs\k1024_metaball_canonical_siren_fp32_1000.json"
$resultRoot = Join-Path $caseRoot "results\full_fp32_k1024_1000_canonical_siren"
$inputDir = Join-Path $resultRoot "inputs"
$runDir = Join-Path $resultRoot "run_001"
$checkpointDir = Join-Path $runDir "checkpoints"
$statePath = Join-Path $resultRoot "campaign_state.json"
New-Item -ItemType Directory -Force -Path $inputDir, $runDir, $checkpointDir | Out-Null

function Write-State([string]$status, [int]$epoch, [string]$message) {
    [ordered]@{
        status = $status; current_epoch = $epoch; target_epochs = 1000
        modes = 1024; method = "native CUDA Jet3 + strict canonical SIREN"
        precision = "all float32"; updated_at = (Get-Date).ToString("o")
        message = $message; run_dir = $runDir
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

try {
    $initial = Join-Path $inputDir "initial_epoch0000.npz"
    $validation = Join-Path $inputDir "fixed_validation_scipy_sobol_v1.npz"
    if (-not (Test-Path -LiteralPath $initial)) {
        Write-State "preparing" 0 "creating strict canonical SIREN initialization"
        & $python (Join-Path $caseRoot "scripts\prepare_metaball_run.py") --config $config --output-dir $inputDir
        if ($LASTEXITCODE -ne 0) { throw "input preparation failed: $LASTEXITCODE" }
    }
    if (-not $SkipPreflight -and -not (Test-Path (Join-Path $resultRoot "preflight_complete.json"))) {
        $preflight = Join-Path $resultRoot "preflight_k1024"
        Write-State "preflight" 0 "one-epoch K=1024 canonical SIREN gate"
        & $python (Join-Path $caseRoot "scripts\train_metaball.py") --resume $initial --run-dir $preflight --validation-pack $validation --self-test-epochs 1 --self-test-points 4096
        if ($LASTEXITCODE -ne 0) { throw "preflight failed: $LASTEXITCODE" }
        $line = Get-Content (Join-Path $preflight "hardcoded_train.log") | Select-String "self-test epoch=.*rank=" | Select-Object -Last 1
        if (-not $line -or $line.Line -notmatch "rank=(\d+)") { throw "preflight rank missing" }
        $rank = [int]$Matches[1]
        if ($rank -lt 1) { throw "preflight produced zero effective rank" }
        # Do not import the spatial-SIREN rank>=512 selection gate here: a low
        # initial rank is a measured outcome of the strict canonical arm and
        # must not trigger method-changing retuning.
        [ordered]@{complete=$true; effective_raw_rank=$rank; selection_gate="none; strict-method diagnostic only"; completed_at=(Get-Date).ToString("o")} | ConvertTo-Json | Set-Content (Join-Path $resultRoot "preflight_complete.json") -Encoding UTF8
    }
    $latest = Join-Path $checkpointDir "hardcoded_latest.npz"
    $resume = if (Test-Path $latest) { $latest } else { $initial }
    $epoch = 0
    $sidecar = [IO.Path]::ChangeExtension($resume, ".json")
    if (Test-Path $sidecar) { $epoch = [int]((Get-Content -Raw $sidecar | ConvertFrom-Json).epoch) }
    Write-State "training" $epoch "training from latest canonical-only checkpoint"
    & $python (Join-Path $caseRoot "scripts\train_metaball.py") --resume $resume --run-dir $runDir --validation-pack $validation
    if ($LASTEXITCODE -ne 0) { throw "training failed: $LASTEXITCODE" }
    Write-State "evaluating" 1000 "rendering canonical SIREN modes"
    & $python (Join-Path $caseRoot "scripts\evaluate_metaball.py") --run-dir $runDir
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed: $LASTEXITCODE" }
    Write-State "awaiting_report" 1000 "training complete; report integration pending"
} catch {
    Write-State "failed" 0 $_.Exception.Message
    throw
}
