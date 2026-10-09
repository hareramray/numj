# Builds the native library (all profiles) and the Java classes. Requires scripts/bootstrap.ps1 first.
#   build/native/numj.dll        strict  (default): -O3 -march=x86-64-v3 (AVX2), no FP contraction
#   build/native/numj-sse2.dll   strict semantics, baseline x86-64 (SSE2, 128-bit vectors)
#   build/native/numj-novec.dll  strict semantics, AVX2 target, auto-vectorization disabled
#   build/native/numj-fast.dll   OPT-IN relaxed FP: -ffast-math (FMA contraction, reassociation,
#                                reciprocal division, assumes no NaN/Inf). Not used unless selected.
# All DLLs are linked statically (libgfortran/libgomp/libgcc inside) so they depend only on Windows DLLs.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')

$Common  = @('-std=f2018', '-Wall', '-Wextra', '-Wno-compare-reals', '-fopenmp', '-frecursive')
$Profiles = [ordered]@{
  'numj'       = @('-O3', '-march=x86-64-v3', '-ffp-contract=off')
  'numj-sse2'  = @('-O3', '-march=x86-64',    '-ffp-contract=off')
  'numj-novec' = @('-O3', '-march=x86-64-v3', '-ffp-contract=off', '-fno-tree-vectorize', '-fno-tree-slp-vectorize')
  'numj-fast'  = @('-O3', '-march=x86-64-v3', '-ffast-math')
}

$Native = Join-Path $Root 'build\native'
New-Item -ItemType Directory -Force $Native | Out-Null
$Src = Join-Path $Root 'native\numj_kernels.f90'
foreach ($name in $Profiles.Keys) {
  $obj = Join-Path $Root "build\obj\$name"
  New-Item -ItemType Directory -Force $obj | Out-Null
  $flags = $Common + $Profiles[$name]
  Write-Host "[fortran] $name : $($flags -join ' ')"
  & gfortran @flags -J $obj -c $Src -o "$obj\numj_kernels.o"
  if ($LASTEXITCODE -ne 0) { throw "gfortran compile failed ($name)" }
  & gfortran -shared -static -fopenmp -o "$Native\$name.dll" "$obj\numj_kernels.o"
  if ($LASTEXITCODE -ne 0) { throw "gfortran link failed ($name)" }
}
# Vectorization report for the default profile (kept for inspection).
& gfortran @Common @($Profiles['numj']) -J "$Root\build\obj\numj" -fopt-info-vec-optimized="$Root\build\vec_report.txt" -c $Src -o "$Root\build\obj\numj\report.o"

function Compile-Java([string]$out, [string[]]$srcDirs, [string]$cp) {
  $files = @(foreach ($d in $srcDirs) { Get-ChildItem -Recurse -Filter *.java (Join-Path $Root $d) | ForEach-Object FullName })
  New-Item -ItemType Directory -Force $out | Out-Null
  $jargs = @('--release', '25', '-Xlint:all', '-Werror', '-d', $out)
  if ($cp) { $jargs += @('-cp', $cp) }
  Write-Host "[javac] $out ($(@($files).Count) files)"
  & javac @jargs @files
  if ($LASTEXITCODE -ne 0) { throw "javac failed for $out" }
}
# Project-relative paths from here on (shorter, location-independent class paths).
Set-Location $Root
$Main = 'build\classes'
$Bench = 'build\bench-classes'
Compile-Java $Main @('java\src') $null
Compile-Java $Bench @('java\bench') $Main
Compile-Java 'build\test-classes' @('java\test') "$Main;$Bench"
Compile-Java 'build\example-classes' @('java\example') $Main
Write-Host '[ok] build complete'
