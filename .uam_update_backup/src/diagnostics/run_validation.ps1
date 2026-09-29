<#
Runs the reorganized UAM verification workflow from anywhere in PowerShell.
Place this script in <project>/src/diagnostics/ and run it from the project root.
Run:  & .\src\diagnostics\run_validation.ps1
Full: & .\src\diagnostics\run_validation.ps1 -Full
#>
param(
    [int] $Seed = 60,
    [switch] $Full
)
$ErrorActionPreference = 'Stop'
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $ProjectDir
Write-Host "Running tests from $ProjectDir"

python -m compileall -q .\src
if ($LASTEXITCODE -ne 0) { throw 'Python syntax compilation failed' }

python -m src.diagnostics.check_data_paths --load
if ($LASTEXITCODE -ne 0) { throw 'Census/ACS path and loading check failed' }

python -m pytest -q .\src\tests
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed' }

$NextSeed = $Seed + 1
python -m src.tests.validate_initial_fleet --solver highs --gap 1e-7 --time-limit 120 --seeds $Seed $NextSeed
if ($LASTEXITCODE -ne 0) { throw 'Detailed/projected validation failed' }

if ($Full) {
    python -m src.experiment_runner --seed $Seed --solver highs --gap 1e-6 --time-limit 120
    if ($LASTEXITCODE -ne 0) { throw 'Development experiment failed' }
    python -m src.diagnostics.fleet_sensitivity --seed $Seed --caps 0 6 12 --costs 30 60 120 --solver highs --gap 1e-6 --time-limit 120
    if ($LASTEXITCODE -ne 0) { throw 'Fleet sensitivity failed' }
}
Write-Host 'Completed requested validation steps.'
