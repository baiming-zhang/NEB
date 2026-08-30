param(
    [int]$Epochs = 200,
    [int]$MaxRetries = 10
)

$ErrorActionPreference = 'Stop'
$caseRoot = Split-Path -Parent $PSScriptRoot
$python = 'C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe'
$runner = Join-Path $PSScriptRoot 'run_siren_sweep.py'
$campaign = Join-Path $caseRoot ("results\siren_frequency_cap_k64\epochs_{0}" -f $Epochs)
New-Item -ItemType Directory -Force -Path $campaign | Out-Null
$stdout = Join-Path $campaign 'launcher.stdout.log'
$stderr = Join-Path $campaign 'launcher.stderr.log'
$processEnvironment = [System.Environment]::GetEnvironmentVariables()
$canonicalPath = [string]$processEnvironment['Path']
[System.Environment]::SetEnvironmentVariable(
    'PATH', $null, [System.EnvironmentVariableTarget]::Process
)
[System.Environment]::SetEnvironmentVariable(
    'Path', $canonicalPath, [System.EnvironmentVariableTarget]::Process
)
$process = Start-Process -FilePath $python `
    -ArgumentList @(
        $runner,
        '--epochs', $Epochs,
        '--max-retries', $MaxRetries
    ) `
    -WorkingDirectory (Resolve-Path (Join-Path $caseRoot '..\..\..')) `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru
@{
    process_id = $process.Id
    started_at = (Get-Date).ToString('o')
    modes = 64
    epochs = $Epochs
    early_stopping_enabled = $false
    runner = $runner
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $campaign 'launcher_process.json')
$process.Id
