$ErrorActionPreference = 'Stop'
$previousLcAll = $env:LC_ALL
$previousLang = $env:LANG
Push-Location $PSScriptRoot
try {
    $env:LC_ALL = 'C'
    $env:LANG = 'C'
    $compiler = Get-Command latexmk -ErrorAction Stop
    New-Item -ItemType Directory -Path build -Force | Out-Null
    & $compiler.Source -pdf -recorder -interaction=nonstopmode -halt-on-error -outdir=build main.tex *> build/compile-output.txt
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath build/compile-output.txt -Tail 50
        throw "LaTeX compilation failed ($LASTEXITCODE). See build/compile-output.txt."
    }
    $buildLog = Get-Content -LiteralPath build/main.log -Raw
    if ($buildLog -notmatch 'Output written on .*\((\d+) pages?,') {
        throw 'The compiler did not confirm a completed PDF.'
    }
    $pageCount = [int]$Matches[1]
    if ($buildLog -match '(?m)^!|LaTeX Warning: (?:There were undefined references|Label\(s\) may have changed)|Package natbib Warning:.*undefined') {
        throw 'The build contains unresolved errors or references; inspect build/main.log.'
    }
    Copy-Item -LiteralPath build/main.pdf -Destination Neural_Eigenfluid_Bases.pdf -Force
    Write-Output "Compiled and updated Neural_Eigenfluid_Bases.pdf ($pageCount pages)."
} finally {
    $env:LC_ALL = $previousLcAll
    $env:LANG = $previousLang
    Pop-Location
}
