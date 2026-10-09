# Packaging prototype (nothing is published): builds, under build\dist,
#   numj-<version>.jar                         Java API only (pure Java, no native code)
#   numj-natives-windows-x86_64-<version>.jar  numj/native/windows-x86_64/x86-64-v3/numj.dll  (AVX2/FMA build)
#                                              numj/native/windows-x86_64/baseline/numj.dll   (SSE2 build)
#   numj-<version>-sources.jar                 sources (Maven Central requires sources + javadoc jars)
# then runs the example from the jars alone, in an empty directory, to prove that the library is found as a
# classpath resource (no build\native, no -Dnumj.library). Requires scripts/build.ps1 first.
param([string]$Version = '0.2.0-SNAPSHOT')
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Root
$Dist = 'build\dist'
if (Test-Path $Dist) { Remove-Item -Recurse -Force $Dist }
New-Item -ItemType Directory -Force "$Dist\natives\numj\native\windows-x86_64\x86-64-v3", "$Dist\natives\numj\native\windows-x86_64\baseline", "$Dist\javadoc" | Out-Null
Copy-Item 'build\native\numj.dll' "$Dist\natives\numj\native\windows-x86_64\x86-64-v3\numj.dll"
Copy-Item 'build\native\numj-sse2.dll' "$Dist\natives\numj\native\windows-x86_64\baseline\numj.dll"

& jar --create --file "$Dist\numj-$Version.jar" -C 'build\classes' .
if ($LASTEXITCODE -ne 0) { throw 'jar (api) failed' }
& jar --create --file "$Dist\numj-natives-windows-x86_64-$Version.jar" -C "$Dist\natives" .
if ($LASTEXITCODE -ne 0) { throw 'jar (natives) failed' }
& jar --create --file "$Dist\numj-$Version-sources.jar" -C 'java\src' .
if ($LASTEXITCODE -ne 0) { throw 'jar (sources) failed' }
$srcs = @(Get-ChildItem -Recurse -Filter *.java 'java\src' | ForEach-Object FullName)
& javadoc -quiet -Xdoclint:none -d "$Dist\javadoc" @srcs | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'javadoc failed' }
& jar --create --file "$Dist\numj-$Version-javadoc.jar" -C "$Dist\javadoc" .
Get-ChildItem $Dist -Filter *.jar | ForEach-Object { Write-Host ("[jar] {0,-48} {1,10:N0} bytes" -f $_.Name, $_.Length) }

# Smoke test from an empty directory: only the two jars and the example classes on the class path.
$smoke = Join-Path $Root "$Dist\smoke"
New-Item -ItemType Directory -Force $smoke | Out-Null
$cp = (@("$Root\$Dist\numj-$Version.jar", "$Root\$Dist\numj-natives-windows-x86_64-$Version.jar", "$Root\build\example-classes")) -join ';'
Push-Location $smoke
try {
  foreach ($level in @('auto', 'baseline')) {
    $props = @('--enable-native-access=ALL-UNNAMED')
    if ($level -ne 'auto') { $props += "-Dnumj.cpu=$level" }
    $out = & java @props -cp $cp Example 2>&1
    if ($LASTEXITCODE -ne 0) { $out | Select-Object -Last 10; throw "example from jars failed (cpu=$level)" }
    $lib = & java @props -cp $cp numj.NativeInfo 2>&1
    Write-Host "[smoke] cpu=$level -> $lib"
  }
} finally {
  Pop-Location
}
Write-Host "[ok] packaging prototype in $Dist (not published)"
