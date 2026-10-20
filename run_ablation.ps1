param(
    [ValidateSet('baseline', 'single', 'main', 'unweighted', 'all')]
    [string]$Variant = 'main',
    [int]$Batch = 8,
    [string]$Device = '0',
    [int]$Seed = 0,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$experiments = @(
    @{ Id='baseline'; Model='baseline'; Name='b0_yolo26n_baseline_300e' },
    @{ Id='single'; Model='direction4_single'; Name='b1_direction4_single_300e' },
    @{ Id='main'; Model='direction4'; Name='b2_direction4_bifpn_300e' },
    @{ Id='unweighted'; Model='direction4_unweighted'; Name='b3_direction4_unweighted_300e' }
)
foreach ($experiment in $experiments) {
    if ($Variant -ne 'all' -and $Variant -ne $experiment.Id) { continue }
    $runName = $experiment.Name
    if ($Seed -ne 0) { $runName += "_seed$Seed" }
    $arguments = @('--model', (Join-Path $PSScriptRoot "configs/models/yolo26n_$($experiment.Model).yaml"),
        '--epochs', '300', '--imgsz', '640', '--batch', $Batch, '--device', $Device,
        '--workers', '4', '--optimizer', 'SGD', '--lr0', '0.01', '--lrf', '0.01',
        '--momentum', '0.937', '--weight-decay', '0.0005', '--warmup-epochs', '3',
        '--patience', '100', '--mosaic', '1', '--close-mosaic', '10', '--seed', $Seed,
        '--name', $runName)
    if ($DryRun) { $arguments += '--dry-run' }
    & (Join-Path $PSScriptRoot 'run_train.ps1') @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
