[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug')]
    [string] $Configuration = 'Release',
    [string] $CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8',
    [ValidatePattern('^sm_[0-9]+$')]
    [string] $Architecture = 'sm_89',
    [switch] $RunSelfTest,
    [switch] $RunMemoryDryRun,
    [UInt64] $Seed = 7640891576956012809
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$nativeRoot = $PSScriptRoot
$sourceFiles = @(
    (Join-Path $nativeRoot 'src\main.cu'),
    (Join-Path $nativeRoot 'src\trainer.cu')
)
$includeDir = Join-Path $nativeRoot 'include'
$buildDir = Join-Path $nativeRoot 'build'
$executable = Join-Path $buildDir 'eigenfluid_native.exe'
$nvcc = Join-Path $CudaRoot 'bin\nvcc.exe'

if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) {
    throw "CUDA compiler was not found at '$nvcc'. Install CUDA Toolkit 11.8 or pass -CudaRoot."
}

$nvccVersion = (& $nvcc --version | Out-String)
if ($nvccVersion -notmatch 'release 11\.8') {
    throw "This build is pinned to CUDA 11.8, but '$nvcc' reported:`n$nvccVersion"
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw "vswhere.exe was not found. Install Visual Studio 2019 with Desktop development with C++."
}

$vsInstall = (& $vswhere -version '[16.0,17.0)' -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath | Select-Object -First 1)
if (-not $vsInstall) {
    throw 'Visual Studio 2019 with the x64 C++ toolchain was not found.'
}

$vsDevCmd = Join-Path $vsInstall 'Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $vsDevCmd -PathType Leaf)) {
    throw "Visual Studio environment script was not found at '$vsDevCmd'."
}

# Import the x64 VS2019 build environment into this PowerShell process. This
# keeps nvcc invocation argument-safe and avoids a generated intermediate .bat.
$vsEnvironmentCommand = '"{0}" -no_logo -arch=x64 -host_arch=x64 >nul && set' -f $vsDevCmd
$environmentLines = & $env:ComSpec /d /s /c $vsEnvironmentCommand
if ($LASTEXITCODE -ne 0) {
    throw "VsDevCmd.bat failed with exit code $LASTEXITCODE."
}
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

$hostCompiler = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $hostCompiler) {
    throw 'cl.exe is unavailable after loading the Visual Studio 2019 environment.'
}

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$optimizationFlags = @('-O3', '-lineinfo')
if ($Configuration -eq 'Debug') {
    $optimizationFlags = @('-O0', '-G', '-g')
}

$nvccArguments = @(
    '-std=c++17',
    "-arch=$Architecture",
    '-I', $includeDir,
    $sourceFiles,
    '-o', $executable,
    '-lcublas',
    '-lcusolver',
    '-Xcompiler', '/EHsc,/W4,/nologo'
) + $optimizationFlags

Write-Host "CUDA: $($nvccVersion.Trim().Split([Environment]::NewLine)[-1])"
Write-Host "Visual Studio: $vsInstall"
Write-Host "Target: $Architecture ($Configuration)"
& $nvcc @nvccArguments
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE."
}

Write-Host "Built: $executable"
if ($RunSelfTest) {
    & $executable --self-test --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Self-test failed with exit code $LASTEXITCODE."
    }
}
if ($RunMemoryDryRun) {
    & $executable --memory-dry-run --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Memory dry run failed with exit code $LASTEXITCODE."
    }
}
