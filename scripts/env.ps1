# Dot-source to put the project-local toolchain first on PATH:  . .\scripts\env.ps1
$Root = Split-Path -Parent $PSScriptRoot
$env:JAVA_HOME = Join-Path $Root '.tools\jdk'
$env:PATH = (Join-Path $Root '.tools\jdk\bin') + ';' + (Join-Path $Root '.tools\mingw64\bin') + ';' + $env:PATH
$NumjPython = Join-Path $Root '.venv\Scripts\python.exe'
$NumjJavaOpts = @('--enable-native-access=ALL-UNNAMED')
