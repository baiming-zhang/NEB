param(
    [Parameter(Mandatory = $true)]
    [string]$Worktree,
    [string]$RunName = "run_001"
)

$ErrorActionPreference = "Stop"
$case = Join-Path $Worktree "cases\2D_airfoil"
$config = Join-Path $case "configs\k1024_cinf_sobol_5000_fp16_pure_vorticity_rayleigh.json"
$inputs = Join-Path $case "reference_inputs\k1024_cinf_sobol_5000_fp16_pure_vorticity_rayleigh"
$run = Join-Path $case "results\k1024_cinf_sobol_5000_fp16_pure_vorticity_rayleigh\$RunName"
$trainer = Join-Path $case "scripts\train_c2_stokes_certified.py"
$checkpoint = Join-Path $inputs "initial_epoch0000.npz"
$validation = Join-Path $inputs "fixed_validation_scipy_sobol_v1.npz"

if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "Missing config: $config"
}
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "Missing epoch-0 checkpoint: $checkpoint"
}
if (-not (Test-Path -LiteralPath $validation -PathType Leaf)) {
    throw "Missing fixed validation pack: $validation"
}
if (Test-Path -LiteralPath $run) {
    throw "Fresh run directory already exists: $run"
}

$matches = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine.Contains("train_c2_stokes_certified.py") -and
    $_.CommandLine.Contains("k1024_cinf_sobol_5000_fp16_pure_vorticity_rayleigh")
}
if ($matches) {
    throw "Matching continuous-vorticity trainer is already running"
}

New-Item -ItemType Directory -Path $run | Out-Null
$stdout = Join-Path $run "launcher.stdout.log"
$stderr = Join-Path $run "launcher.stderr.log"
$arguments = @(
    "-u",
    $trainer,
    "--resume", $checkpoint,
    "--run-dir", $run,
    "--validation-pack", $validation
)
$process = Start-Process -FilePath "python" -ArgumentList $arguments `
    -WorkingDirectory $Worktree -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr

$launch = @{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    worktree = $Worktree
    config = $config
    initial_checkpoint = $checkpoint
    fixed_validation_pack = $validation
    run = $run
    stdout = $stdout
    stderr = $stderr
}
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run "launch.json") `
    -Encoding UTF8
$process.Id
