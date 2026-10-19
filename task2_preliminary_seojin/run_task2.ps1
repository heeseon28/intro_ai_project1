param(
    [ValidateSet('full', 'smoke', 'preflight')][string]$Mode = 'full',
    [string]$DataRoot,
    [string]$Python,
    [ValidateSet('auto', 'cuda', 'mps', 'cpu')][string]$Device = 'auto',
    [string]$RunName = 'baseline_seed0',
    [int]$Threads = 4,
    [int]$Workers = 0,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $DataRoot) { $DataRoot = Join-Path (Split-Path -Parent $repoRoot) 'VMMRdb' }
if (-not $Python) {
    $localPython = Join-Path $repoRoot '.venv-project-runner/Scripts/python.exe'
    if (Test-Path -LiteralPath $localPython) { $Python = $localPython }
    else { $Python = 'python' }
}
$env:PYTHONIOENCODING = 'utf-8'
if ($Mode -eq 'preflight') {
    & $Python (Join-Path $PSScriptRoot 'preflight.py') --data-root $DataRoot
} else {
    if ($Mode -eq 'full') {
        $processFile = Join-Path $PSScriptRoot "results/$RunName/process.json"
        if (Test-Path -LiteralPath $processFile) {
            $processRecord = Get-Content -LiteralPath $processFile -Raw -Encoding UTF8 | ConvertFrom-Json
            if (Get-Process -Id $processRecord.pid -ErrorAction SilentlyContinue) {
                throw "Training process $($processRecord.pid) is already active. Do not start a second writer."
            }
        }
    }
    $runArgs = @((Join-Path $PSScriptRoot 'train_baseline.py'), '--data-root', $DataRoot,
        '--device', $Device, '--run-name', $RunName, '--threads', $Threads, '--workers', $Workers)
    if ($Mode -eq 'smoke') { $runArgs += '--smoke' }
    if ($Resume) { $runArgs += '--resume' }
    & $Python @runArgs
}
exit $LASTEXITCODE
