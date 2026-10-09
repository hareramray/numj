# Differential tests against the pinned NumPy: generates reference results with difftest/gen_numpy_cases.py into
# build\difftest, then replays them with numj.DiffTests. Requires scripts/bootstrap.ps1 and scripts/build.ps1 first.
param([string]$Library = 'build\native\numj.dll')
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Root
if (Test-Path 'build\difftest') { Remove-Item -Recurse -Force 'build\difftest' }
& $NumjPython -I 'difftest\gen_numpy_cases.py' 'build\difftest'
if ($LASTEXITCODE -ne 0) { throw 'difftest generation failed' }
& java @NumjJavaOpts "-Dnumj.library=$Library" -cp 'build\classes;build\bench-classes;build\test-classes' numj.DiffTests 'build\difftest'
exit $LASTEXITCODE
