# Runs the correctness suites against build/native/numj.dll (or -Library <path>). Exit code != 0 on failure.
#   NumJTests     fused kernels, numerical contract, memory model (numj 0.1 suite)
#   NDArrayTests  n-d arrays: views, broadcasting, elementwise, reductions, overlap, lifetime, threads
#   DiffTests     differential tests against NumPy (needs build\difftest from scripts\difftest.ps1; skipped otherwise)
param([string]$Library = 'build\native\numj.dll', [string[]]$Suites = @('NumJTests', 'NDArrayTests', 'DiffTests'))
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Root
$Suites = @($Suites | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$failed = @()
foreach ($s in $Suites) {
  if ($s -eq 'DiffTests' -and -not (Test-Path 'build\difftest\cases.json')) {
    Write-Host "[skip] DiffTests: build\difftest\cases.json not found (run scripts\difftest.ps1)"
    continue
  }
  Write-Host "=== $s ($Library)"
  & java @NumjJavaOpts "-Dnumj.library=$Library" -Xss4m -cp 'build\classes;build\bench-classes;build\test-classes' "numj.$s"
  if ($LASTEXITCODE -ne 0) { $failed += $s }
}
if ($failed.Count -gt 0) { Write-Host "[FAIL] $($failed -join ', ')"; exit 1 }
Write-Host '[ok] all suites passed'
exit 0
