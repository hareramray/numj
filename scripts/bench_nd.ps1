# Milestone-1 benchmarks (n-d arrays): JMH for numj and plain Java, bench/numpy_nd_bench.py for NumPy, NumExpr
# and Numba. Pinned to the P-cores like scripts/bench.ps1. Writes results\nd\*.json|csv|log and environment.json;
# bench\report_nd.py turns them into results\nd\RESULTS.md.
#   -Jmh      regex of JMH benchmarks to run (default: all numj.*Bench); '' to skip JMH
#   -Python   run the Python suite (default on); -NoPython to skip
#   -Tag      file name suffix, e.g. 'decision' (default: 'main')
#   -Mask     CPU affinity mask (default 0xFF = logical CPUs 0-7 = the four P-cores of an i5-13420H)
param([string]$Jmh = 'numj\..*Bench', [switch]$NoPython, [string]$Tag = 'main', [int]$Mask = 0xFF,
      [string]$OutDir = 'results\nd', [string[]]$JmhExtra = @())
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Root
$JmhExtra = @($JmhExtra | ForEach-Object { $_ -split ',' } | Where-Object { $_ })   # accept "a,b" from -File
New-Item -ItemType Directory -Force $OutDir | Out-Null
$JavaExe = Join-Path $Root '.tools\jdk\bin\java.exe'
$JmhCp = (@('build\classes', 'build\bench-classes', 'build\jmh-classes') +
          @(Get-ChildItem '.tools\jmh' -Filter *.jar | ForEach-Object { ".tools\jmh\$($_.Name)" })) -join ';'

function Invoke-Pinned([string]$exe, [string[]]$argv, [string]$log) {
  Write-Host "[run] $log"
  $p = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  $null = $p.Handle
  $p.ProcessorAffinity = [IntPtr]$Mask        # JMH forks inherit the affinity of this process
  $p.WaitForExit()
  if ($p.ExitCode -ne 0) { Get-Content "$log.err" | Select-Object -Last 20; throw "$exe failed ($($p.ExitCode))" }
}

if ($Jmh) {
  Invoke-Pinned $JavaExe (@('--enable-native-access=ALL-UNNAMED', '-cp', $JmhCp, 'org.openjdk.jmh.Main', $Jmh,
      '-rf', 'json', '-rff', "$OutDir\jmh_$Tag.json") + $JmhExtra) "$OutDir\jmh_$Tag.log"
}
if (-not $NoPython) {
  Invoke-Pinned $NumjPython @('-I', 'bench\numpy_nd_bench.py', '--out', "$OutDir\python_$Tag.csv") "$OutDir\python_$Tag.log"
}

# Environment record (hardware, OS, power, toolchain, compiler flags of the native build).
$cpu = Get-CimInstance Win32_Processor
$os = Get-CimInstance Win32_OperatingSystem
$env = [ordered]@{
  date          = (Get-Date).ToString('s')
  cpu           = $cpu.Name.Trim()
  cores         = $cpu.NumberOfCores
  logical       = $cpu.NumberOfLogicalProcessors
  max_mhz       = $cpu.MaxClockSpeed
  l2_kb         = $cpu.L2CacheSize
  l3_kb         = $cpu.L3CacheSize
  ram_gb        = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
  os            = "$($os.Caption) $($os.Version) build $($os.BuildNumber)"
  power_plan    = ((powercfg /getactivescheme) -join ' ').Trim()
  on_ac_power   = ((Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus -eq 2)
  affinity_mask = ('0x{0:X}' -f $Mask)
  java          = ((cmd /c 'java -version 2>&1') | ForEach-Object { "$_" }) -join ' | '
  jmh           = 'JMH 1.37'
  gfortran      = (& gfortran --version | Select-Object -First 1)
  native_build  = ((& java --enable-native-access=ALL-UNNAMED -cp 'build\classes;build\bench-classes' numj.bench.Bench --suite fingerprints) |
                   Where-Object { $_ -like '# native*' -or $_ -like '# build*' -or $_ -like '# java*' }) -join ' | '
  fortran_flags = (Get-Content 'scripts\build.ps1' | Select-String "^\s+'numj(-\w+)?'\s+=" | ForEach-Object { $_.Line.Trim() })
}
$env | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 "$OutDir\environment_$Tag.json"
Write-Host "[ok] results in $OutDir"
