# Maven Central release of numj (see docs/RELEASING.md).
#
#   scripts\release.ps1             DRY RUN: build, all test suites on both shipped native builds, stage the DLLs,
#                                   mvn verify without signing, check jar contents, smoke-test the jars from an
#                                   empty directory. Uploads nothing.
#   scripts\release.ps1 -Publish    the same, then signs (GPG) and uploads to the Sonatype Central Portal with
#                                   autoPublish=false: the deployment is validated and then WAITS in
#                                   https://central.sonatype.com/publishing/deployments until you press "Publish"
#                                   (or "Drop"). A published version can never be changed or deleted.
#   -SkipBuild                      reuse build\ as it is (still runs the tests)
#
# -Publish needs: a Central Portal user token in %USERPROFILE%\.m2\settings.xml (server id "central"), and a GPG
# secret key whose public key is on a public key server. It also refuses to run from a dirty working tree.
param([switch]$Publish, [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Root
$Mvn = Join-Path $Root '.tools\maven\bin\mvn.cmd'
if (-not (Test-Path $Mvn)) { throw 'Maven not found in .tools\maven (run scripts\bootstrap.ps1)' }
# gpg: PATH first, else the copy that ships with Git for Windows (not on PowerShell's PATH by default)
$Gpg = (Get-Command gpg -ErrorAction SilentlyContinue).Source
if (-not $Gpg) {
  foreach ($c in @("$env:ProgramFiles\Git\usr\bin\gpg.exe", "${env:ProgramFiles(x86)}\Git\usr\bin\gpg.exe")) {
    if (Test-Path $c) { $Gpg = $c; break }
  }
}
[xml]$pom = Get-Content 'pom.xml'
$Version = $pom.project.version
Write-Host "[release] numj $Version ($(if ($Publish) { 'PUBLISH' } else { 'dry run' }))"

if ($Publish) {
  $dirty = & git status --porcelain
  if ($dirty) { throw 'working tree is not clean; commit or stash first' }
  if (-not (Test-Path (Join-Path $env:USERPROFILE '.m2\settings.xml'))) { throw 'no %USERPROFILE%\.m2\settings.xml with the "central" server token (see docs/RELEASING.md)' }
  if (-not $Gpg) { throw 'gpg not found (install Git for Windows or Gpg4win)' }
  $keys = & $Gpg --list-secret-keys 2>$null
  if (-not $keys) { throw 'no GPG secret key (see docs/RELEASING.md)' }
}

# 1. build and test both native builds that ship (AVX2 and SSE2 baseline)
if (-not $SkipBuild) {
  & powershell -ExecutionPolicy Bypass -File scripts\build.ps1
  if ($LASTEXITCODE -ne 0) { throw 'build failed' }
}
foreach ($lib in @('numj', 'numj-sse2')) {
  & powershell -ExecutionPolicy Bypass -File scripts\test.ps1 -Library "build\native\$lib.dll"
  if ($LASTEXITCODE -ne 0) { throw "tests failed for $lib.dll" }
}

# 2. stage the natives in the layout NativeLoader looks for
$stage = 'build\stage\natives\numj\native\windows-x86_64'
if (Test-Path 'build\stage') { Remove-Item -Recurse -Force 'build\stage' }
New-Item -ItemType Directory -Force "$stage\x86-64-v3", "$stage\baseline" | Out-Null
Copy-Item 'build\native\numj.dll' "$stage\x86-64-v3\numj.dll"
Copy-Item 'build\native\numj-sse2.dll' "$stage\baseline\numj.dll"

# 3. maven: package (+ sign and upload when publishing)
$mvnArgs = @('-B', '-ntp', 'clean')
if ($Publish) { $mvnArgs += @('deploy', "-Dgpg.executable=$Gpg") } else { $mvnArgs += @('verify', '-Dgpg.skip=true') }
& $Mvn @mvnArgs
if ($LASTEXITCODE -ne 0) { throw 'maven failed' }

# 4. check the produced jars
$api = "maven\numj\target\numj-$Version.jar"
$nat = "maven\numj-natives-windows-x86_64\target\numj-natives-windows-x86_64-$Version.jar"
foreach ($f in @($api, "maven\numj\target\numj-$Version-sources.jar", "maven\numj\target\numj-$Version-javadoc.jar", $nat,
                 "maven\numj-natives-windows-x86_64\target\numj-natives-windows-x86_64-$Version-sources.jar",
                 "maven\numj-natives-windows-x86_64\target\numj-natives-windows-x86_64-$Version-javadoc.jar")) {
  if (-not (Test-Path $f)) { throw "missing $f" }
}
$natList = & jar tf $nat
foreach ($e in @('numj/native/windows-x86_64/x86-64-v3/numj.dll', 'numj/native/windows-x86_64/baseline/numj.dll', 'META-INF/NOTICE', 'META-INF/LICENSE')) {
  if ($natList -notcontains $e) { throw "$nat lacks $e" }
}
$apiList = & jar tf $api
if (($apiList | Where-Object { $_ -like '*.dll' }) -or ($apiList -notcontains 'numj/NumJ.class')) { throw "$api has unexpected contents" }
Get-ChildItem maven\*\target\*.jar | ForEach-Object { Write-Host ("[jar] {0,-58} {1,10:N0} bytes" -f $_.Name, $_.Length) }

# 5. smoke test: the example against the two jars only, from an empty directory, both CPU levels
$smoke = Join-Path $Root 'build\stage\smoke'
New-Item -ItemType Directory -Force $smoke | Out-Null
$cp = (@("$Root\$api", "$Root\$nat", "$Root\build\example-classes")) -join ';'
Push-Location $smoke
try {
  foreach ($cpu in @('auto', 'baseline')) {
    $props = @('--enable-native-access=ALL-UNNAMED')
    if ($cpu -ne 'auto') { $props += "-Dnumj.cpu=$cpu" }
    $out = & java @props -cp $cp Example 2>&1
    if ($LASTEXITCODE -ne 0 -or -not ($out -match 'kernel sumSqMulAdd = 678.0')) { $out | Select-Object -Last 10; throw "smoke test failed (cpu=$cpu)" }
    Write-Host "[smoke] cpu=${cpu}: $(& java @props -cp $cp numj.NativeInfo 2>&1)"
  }
} finally {
  Pop-Location
}
if ($Publish) {
  Write-Host "[ok] uploaded numj $Version. Review and press Publish at https://central.sonatype.com/publishing/deployments"
} else {
  Write-Host "[ok] dry run complete for numj $Version (nothing uploaded)"
}
