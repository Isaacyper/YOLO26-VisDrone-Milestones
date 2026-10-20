param(
    [ValidateSet('control', 'dysample')][string]$Variant = 'dysample',
    [ValidateRange(0, 2147483647)][int]$Seed = 0,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$model = if ($Variant -eq 'control') { 'yolo26n_direction46_single.yaml' } else { 'yolo26n_direction46_single_dysample.yaml' }
$name = if ($Variant -eq 'control') { 'ds_f2_control_300e' } else { 'ds_f2_dysample_300e' }
$arguments = @('--model', (Join-Path $PSScriptRoot "configs/models/$model"),
    '--weights', (Join-Path $PSScriptRoot 'weights/yolo26n.pt'),
    '--epochs', '300', '--imgsz', '640', '--batch', '8', '--workers', '0',
    '--box-iou', 'ciou', '--seed', $Seed, '--name', "${name}_seed$Seed")
if ($DryRun) { $arguments += '--dry-run' }
& (Join-Path $PSScriptRoot 'run_train.ps1') @arguments
exit $LASTEXITCODE
