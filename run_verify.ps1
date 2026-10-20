param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
$python = 'D:\Driver\python\envs\alss\python.exe'
$arguments = @('-B', '-X', 'utf8', (Join-Path $PSScriptRoot 'scripts/verify.py'))
if ($Smoke) { $arguments += '--smoke' }
& $python @arguments
exit $LASTEXITCODE
