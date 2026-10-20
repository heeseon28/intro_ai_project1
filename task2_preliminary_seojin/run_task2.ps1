param(
    [string]$DataRoot = "",
    [ValidateSet("auto", "cuda", "mps", "cpu")][string]$Device = "auto",
    [string]$RunName = "baseline_seed0_seojin",
    [int]$Workers = 0,
    [int]$BatchSize = 64,
    [string]$Python = "python",
    [switch]$CheckOnly,
    [switch]$Smoke,
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
if ($CheckOnly) {
    $RunArgs = @((Join-Path $PSScriptRoot "preflight.py"))
} else {
    $RunArgs = @((Join-Path $PSScriptRoot "train_baseline.py"),
        "--device", $Device, "--run-name", $RunName,
        "--workers", "$Workers", "--batch-size", "$BatchSize")
    if ($Smoke) { $RunArgs += "--smoke" }
    if ($Resume) { $RunArgs += "--resume" }
}
if ($DataRoot) { $RunArgs += @("--data-root", $DataRoot) }
& $Python @RunArgs
exit $LASTEXITCODE
