param(
    [ValidateSet('control', 'fusion', 'single', 'all')][string]$Variant = 'fusion',
    [int]$Seed = 0,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$experiments = @(
    @{Id='control'; Model='direction6'; Name='f0_direction6_control_300e'},
    @{Id='fusion'; Model='direction46'; Name='f1_direction46_bifpn_300e'},
    @{Id='single'; Model='direction46_single'; Name='f2_direction46_single_300e'}
)
foreach ($experiment in $experiments) {
    # Keep the original all queue limited to F0/F1; F2 is an explicit follow-up.
    if ($Variant -eq 'all' -and $experiment.Id -eq 'single') { continue }
    if ($Variant -ne 'all' -and $Variant -ne $experiment.Id) { continue }
    $arguments = @('--model', (Join-Path $PSScriptRoot "configs/models/yolo26n_$($experiment.Model).yaml"),
        '--weights', (Join-Path $PSScriptRoot 'weights/yolo26n.pt'),
        '--epochs', '300', '--imgsz', '640', '--batch', '8', '--workers', '0',
        '--seed', $Seed, '--name', "$($experiment.Name)_seed$Seed")
    if ($DryRun) { $arguments += '--dry-run' }
    & (Join-Path $PSScriptRoot 'run_train.ps1') @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
