param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv-build"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResolvedVenvDir = Join-Path $ProjectRoot $VenvDir
$VenvPython = Join-Path $ResolvedVenvDir "Scripts\\python.exe"
$BuildRequirements = Join-Path $ProjectRoot "requirements-build.txt"
$BuildScript = Join-Path $ProjectRoot "build_minimal.py"
$ExePath = Join-Path $ProjectRoot "dist\\OptCNT.exe"

if (-not (Test-Path $ResolvedVenvDir)) {
    Write-Host "Creating virtual environment at $ResolvedVenvDir"
    & $Python -m venv $ResolvedVenvDir
}

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python was not found: $VenvPython"
}

Write-Host "Upgrading packaging tools inside the virtual environment"
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing minimal build dependencies"
& $VenvPython -m pip install -r $BuildRequirements

Write-Host "Building OptCNT with UPX compression"
& $VenvPython $BuildScript

if (-not (Test-Path $ExePath)) {
    throw "Build finished but the executable was not found: $ExePath"
}

$SizeMB = [Math]::Round((Get-Item $ExePath).Length / 1MB, 2)
Write-Host "Build complete: $ExePath"
Write-Host "Executable size: $SizeMB MB"
