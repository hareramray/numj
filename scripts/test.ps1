# Runs the correctness tests against build/native/numj.dll (or -Library <path>). Exit code != 0 on failure.
param([string]$Library = 'build\native\numj.dll')
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Root
& java @NumjJavaOpts "-Dnumj.library=$Library" -cp 'build\classes;build\bench-classes;build\test-classes' numj.NumJTests
exit $LASTEXITCODE
