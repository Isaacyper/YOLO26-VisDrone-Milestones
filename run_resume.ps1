param([Parameter(Mandatory=$true)][string]$Checkpoint, [switch]$DryRun)
$ErrorActionPreference = 'Stop'
$resolved = (Resolve-Path -LiteralPath $Checkpoint).Path
$arguments = @('--model', $resolved, '--resume')
if ($DryRun) { $arguments += '--dry-run' }
& (Join-Path $PSScriptRoot 'run_train.ps1') @arguments
exit $LASTEXITCODE
