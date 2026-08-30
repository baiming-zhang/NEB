param(
    [string]$RunDir = "",
    [int]$MaximumRestarts = 20
)

$ErrorActionPreference = "Stop"
$CaseRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Config = Join-Path $CaseRoot "configs\k64_three_sphere_connected_siren_dirichlet_500.json"
if (-not $RunDir) {
    $RunDir = Join-Path $CaseRoot "results\k64_siren_dirichlet_e500\run_001"
}
$RunDir = [System.IO.Path]::GetFullPath($RunDir)
$Checkpoint = Join-Path $RunDir "checkpoints\latest.pt"

for ($Attempt = 0; $Attempt -le $MaximumRestarts; $Attempt++) {
    $Arguments = @(
        (Join-Path $CaseRoot "scripts\train_validate.py"),
        "--config", $Config,
        "--run-dir", $RunDir
    )
    if (Test-Path -LiteralPath $Checkpoint) {
        $Arguments += @("--resume", $Checkpoint)
    }
    & python @Arguments
    if ($LASTEXITCODE -eq 0) {
        exit 0
    }
    Add-Content -LiteralPath (Join-Path $RunDir "restart.log") -Value (
        "[{0}] trainer exit={1}; restart attempt={2}" -f (Get-Date -Format s), $LASTEXITCODE, ($Attempt + 1)
    )
}
throw "training failed after $MaximumRestarts restart attempts"

