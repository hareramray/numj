# Runs the benchmarks, pinned to the P-cores, and writes results/ (CSV + logs + environment) and RESULTS.md.
#   -Suites  any of: main numpy profiles threads overhead   (default: all)
#   -Quick   short warmup/sampling for smoke tests (numbers not meaningful)
#   -OutDir  output directory (default results; relative to the project root, no spaces)
#   -Mask    CPU affinity mask (default 0xFF = logical CPUs 0-7 = the four P-cores on an i5-13420H;
#            check yours with the CPU-set dump in README).
param([string[]]$Suites = @('main', 'numpy', 'profiles', 'threads', 'overhead'), [switch]$Quick, [int]$Mask = 0xFF, [string]$OutDir = 'results')
$ErrorActionPreference = 'Stop'
$Suites = @($Suites | ForEach-Object { $_ -split ',' } | Where-Object { $_ })   # accept "a,b" from -File
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $Root
$Res = $OutDir
New-Item -ItemType Directory -Force $Res | Out-Null
$Q = if ($Quick) { @('--quick') } else { @() }
$Cp = 'build\classes;build\bench-classes'
$JavaExe = Join-Path $Root '.tools\jdk\bin\java.exe'

# Start a process, pin it before it does real work, wait, fail on non-zero exit.
# (Arguments must not contain spaces: Start-Process in PowerShell 5.1 does not quote them.)
function Invoke-Pinned([string]$exe, [string[]]$argv, [string]$log) {
  Write-Host "[run] $log"
  $p = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  $null = $p.Handle                              # needed for ExitCode in Windows PowerShell
  $p.ProcessorAffinity = [IntPtr]$Mask
  $p.WaitForExit()
  if ($p.ExitCode -ne 0) { Get-Content "$log.err" | Select-Object -Last 20; throw "$exe failed ($($p.ExitCode))" }
}

$JavaBase = @('--enable-native-access=ALL-UNNAMED', '-Xms4g', '-Xmx4g', '-XX:+AlwaysPreTouch', '-cp', $Cp)
if ($Suites -contains 'main') {
  Invoke-Pinned $JavaExe ($JavaBase + @('numj.bench.Bench', '--suite', 'main', '--out', "$Res\java_main.csv") + $Q) "$Res\java_main.log"
  Invoke-Pinned $JavaExe ($JavaBase + @('numj.bench.Bench', '--suite', 'fingerprints')) "$Res\java_fingerprints.log"
}
if ($Suites -contains 'numpy') {
  Invoke-Pinned $NumjPython (@('-I', 'bench\numpy_bench.py', '--out', "$Res\numpy_main.csv") + $Q) "$Res\numpy_main.log"
}
if ($Suites -contains 'profiles') {
  foreach ($prof in @('numj', 'numj-sse2', 'numj-novec', 'numj-fast')) {
    Invoke-Pinned $JavaExe (@("-Dnumj.library=build\native\$prof.dll") + $JavaBase + @('numj.bench.Bench', '--suite', 'profiles', '--profile', $prof, '--out', "$Res\java_profile_$prof.csv") + $Q) "$Res\java_profile_$prof.log"
  }
}
if ($Suites -contains 'threads') {
  Invoke-Pinned $JavaExe ($JavaBase + @('numj.bench.Bench', '--suite', 'threads', '--out', "$Res\java_threads.csv") + $Q) "$Res\java_threads.log"
}
if ($Suites -contains 'overhead') {
  Invoke-Pinned $JavaExe ($JavaBase + @('numj.bench.Bench', '--suite', 'overhead', '--out', "$Res\java_overhead_critical.csv") + $Q) "$Res\java_overhead_critical.log"
  Invoke-Pinned $JavaExe (@('-Dnumj.criticalMaxElements=-1') + $JavaBase + @('numj.bench.Bench', '--suite', 'overhead', '--out', "$Res\java_overhead_regular.csv") + $Q) "$Res\java_overhead_regular.log"
}

# Environment record.
$cpu = Get-CimInstance Win32_Processor
$os = Get-CimInstance Win32_OperatingSystem
$env = [ordered]@{
  date          = (Get-Date).ToString('s')
  cpu           = $cpu.Name.Trim()
  cores         = $cpu.NumberOfCores
  logical       = $cpu.NumberOfLogicalProcessors
  l2_kb         = $cpu.L2CacheSize
  l3_kb         = $cpu.L3CacheSize
  ram_gb        = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
  os            = "$($os.Caption) $($os.Version) build $($os.BuildNumber)"
  power_plan    = ((powercfg /getactivescheme) -join ' ').Trim()
  on_ac_power   = ((Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus -eq 2)
  affinity_mask = ('0x{0:X}' -f $Mask)
  java          = ((cmd /c 'java -version 2>&1') | ForEach-Object { "$_" }) -join ' | '
  java_flags    = $JavaBase -join ' '
  gfortran      = (& gfortran --version | Select-Object -First 1)
  fortran_flags = (Get-Content 'scripts\build.ps1' | Select-String "^\s+'numj(-\w+)?'\s+=" | ForEach-Object { $_.Line.Trim() })
}
$env | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 "$Res\environment.json"
if (-not $Quick) {
  & $NumjPython -I 'bench\report.py' $Res
  if ($LASTEXITCODE -ne 0) { throw 'bench\report.py failed' }
}
Write-Host "[ok] results in $Res"
