# Downloads a pinned, project-local toolchain into .tools/ (no admin rights, no system changes).
#   - Eclipse Temurin JDK 25 (LTS)            -> .tools/jdk
#   - WinLibs MinGW-w64 GCC/gfortran (UCRT)   -> .tools/mingw64
#   - Apache Maven 3.9.9 (releases only)      -> .tools/maven
#   - JMH 1.37 jars (Maven Central)           -> .tools/jmh
#   - Python venv with pinned NumPy           -> .venv  (+ NumExpr/Numba from bench/requirements-compare.txt)
# Every archive is verified against a pinned SHA-256 before extraction.
# Re-running is safe: completed steps are skipped.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root  = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root '.tools'

$Pins = @(
  @{ Name   = 'jdk'
     Url    = 'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.4.1%2B1/OpenJDK25U-jdk_x64_windows_hotspot_25.0.4.1_1.zip'
     Sha256 = '00c847d804f4a78e9f04f2683faf14fed898535b177b7fc704486cb0284e9283'
     Probe  = 'bin\java.exe' },
  @{ Name   = 'mingw64'
     Url    = 'https://github.com/brechtsanders/winlibs_mingw/releases/download/16.2.0posix-14.0.0-ucrt-r2/winlibs-x86_64-posix-seh-gcc-16.2.0-mingw-w64ucrt-14.0.0-r2.zip'
     Sha256 = 'd5dbafc4a170e762ca6143151ec918fb9e2c72736fb14cd704abebc6bdd5276a'
     Probe  = 'bin\gfortran.exe' },
  # Apache Maven (only for scripts/release.ps1). SHA-256 of the archive whose SHA-512 matches Apache's published
  # apache-maven-3.9.9-bin.zip.sha512 (8beac8d1...c4418ba).
  @{ Name   = 'maven'
     Url    = 'https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.zip'
     Sha256 = '4ec3f26fb1a692473aea0235c300bd20f0f9fe741947c82c1234cefd76ac3a3c'
     Probe  = 'bin\mvn.cmd' }
)

New-Item -ItemType Directory -Force $Tools | Out-Null
foreach ($p in $Pins) {
  $dest = Join-Path $Tools $p.Name
  if (Test-Path (Join-Path $dest $p.Probe)) { Write-Host "[ok] $($p.Name) already installed"; continue }

  # Each archive gets its own fresh download directory and its own fresh extraction directory.
  $dl  = Join-Path $Tools "downloads\$($p.Name)"
  $tmp = Join-Path $Tools "extract-$($p.Name)"
  foreach ($d in @($dl, $tmp)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d }; New-Item -ItemType Directory $d | Out-Null }
  $zip = Join-Path $dl 'archive.zip'

  Write-Host "[..] downloading $($p.Name)"
  & curl.exe -fL --retry 3 -o $zip $p.Url
  if ($LASTEXITCODE -ne 0) { throw "download failed: $($p.Url)" }
  $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
  if ($hash -ne $p.Sha256) { throw "SHA-256 mismatch for $($p.Name): got $hash expected $($p.Sha256)" }
  Write-Host "[ok] sha256 verified"

  & tar.exe -xf $zip -C $tmp
  if ($LASTEXITCODE -ne 0) { throw "extract failed: $zip" }
  # Both archives contain exactly one top-level directory; move it to the stable name.
  $top = Get-ChildItem $tmp -Directory | Select-Object -First 1
  if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
  Move-Item $top.FullName $dest
  Remove-Item -Recurse -Force $tmp, $dl
  if (-not (Test-Path (Join-Path $dest $p.Probe))) { throw "$($p.Name): $($p.Probe) missing after extraction" }
  Write-Host "[ok] $($p.Name) -> $dest"
}

# JMH 1.37 and its dependencies (Maven Central), for the Java microbenchmarks in java/jmh. Pinned by SHA-256
# (computed on first download after checking Maven Central's published SHA-1).
$JmhJars = @(
  @{ Path = 'org/openjdk/jmh/jmh-core/1.37/jmh-core-1.37.jar'; Sha256 = 'dc0eaf2bbf0036a70b60798c785d6e03a9daf06b68b8edb0f1ba9eb3421baeb3' },
  @{ Path = 'org/openjdk/jmh/jmh-generator-annprocess/1.37/jmh-generator-annprocess-1.37.jar'; Sha256 = '6a5604b5b804e0daca1145df1077609321687734a8b49387e49f10557c186c77' },
  @{ Path = 'net/sf/jopt-simple/jopt-simple/5.0.4/jopt-simple-5.0.4.jar'; Sha256 = 'df26cc58f235f477db07f753ba5a3ab243ebe5789d9f89ecf68dd62ea9a66c28' },
  @{ Path = 'org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar'; Sha256 = '1e56d7b058d28b65abd256b8458e3885b674c1d588fa43cd7d1cbb9c7ef2b308' }
)
$JmhDir = Join-Path $Tools 'jmh'
New-Item -ItemType Directory -Force $JmhDir | Out-Null
foreach ($j in $JmhJars) {
  $dest = Join-Path $JmhDir (Split-Path -Leaf $j.Path)
  if (-not (Test-Path $dest)) {
    & curl.exe -fsSL --retry 3 -o $dest "https://repo1.maven.org/maven2/$($j.Path)"
    if ($LASTEXITCODE -ne 0) { throw "download failed: $($j.Path)" }
  }
  $hash = (Get-FileHash -Algorithm SHA256 $dest).Hash.ToLower()
  if ($hash -ne $j.Sha256) { Remove-Item $dest; throw "SHA-256 mismatch for $($j.Path)" }
}
Write-Host "[ok] JMH jars in $JmhDir"

# Python venv with the pinned NumPy (uv if present, else the stdlib venv + pip).
$venvPy = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
  if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv venv --python 3.13 (Join-Path $Root '.venv')
    & uv pip install --python $venvPy -r (Join-Path $Root 'bench\requirements.txt')
  } else {
    & py -3.13 -m venv (Join-Path $Root '.venv')
    & $venvPy -m pip install -r (Join-Path $Root 'bench\requirements.txt')
  }
  if ($LASTEXITCODE -ne 0) { throw 'venv setup failed' }
}
# Optional comparison baselines (NumExpr, Numba) for bench/numpy_nd_bench.py; NumPy stays at the pinned version.
$cmp = Join-Path $Root 'bench\requirements-compare.txt'
if (Get-Command uv -ErrorAction SilentlyContinue) { & uv pip install --python $venvPy -r $cmp }
else { & $venvPy -m pip install -r $cmp }
if ($LASTEXITCODE -ne 0) { Write-Host '[warn] comparison packages not installed; numpy_nd_bench.py will skip NumExpr/Numba' }
Write-Host "[ok] python venv: $venvPy"
& $venvPy -c "import numpy, sys; print('numpy', numpy.__version__, 'python', sys.version.split()[0])"
