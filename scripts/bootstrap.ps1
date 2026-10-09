# Downloads a pinned, project-local toolchain into .tools/ (no admin rights, no system changes).
#   - Eclipse Temurin JDK 25 (LTS)            -> .tools/jdk
#   - WinLibs MinGW-w64 GCC/gfortran (UCRT)   -> .tools/mingw64
#   - Python venv with pinned NumPy           -> .venv
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
     Probe  = 'bin\gfortran.exe' }
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
Write-Host "[ok] python venv: $venvPy"
& $venvPy -c "import numpy, sys; print('numpy', numpy.__version__, 'python', sys.version.split()[0])"
