$ErrorActionPreference = 'Stop'

$knownPython = 'D:\Driver\python\envs\alss\python.exe'
$python = if ($env:YOLO26_PYTHON) {
    $env:YOLO26_PYTHON
} elseif (Test-Path -LiteralPath $knownPython) {
    $knownPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}
$script = Join-Path $PSScriptRoot 'scripts\compare_results.py'

& $python -B $script @args
exit $LASTEXITCODE
