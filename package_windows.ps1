param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv-build",
    [ValidateSet("slim", "performance")]
    [string]$Profile = "slim"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResolvedVenvDir = Join-Path $ProjectRoot $VenvDir
$VenvPython = Join-Path $ResolvedVenvDir "Scripts\\python.exe"
$BuildRequirements = Join-Path $ProjectRoot "requirements-build.txt"
$BuildScript = Join-Path $ProjectRoot "build_minimal.py"
$ExePath = Join-Path $ProjectRoot "dist\\OptCNT.exe"

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

if (Test-Path $ResolvedVenvDir) {
    Write-Host "Refreshing build virtual environment at $ResolvedVenvDir"
    Remove-Item -LiteralPath $ResolvedVenvDir -Recurse -Force
}

Write-Host "Creating virtual environment at $ResolvedVenvDir"
Invoke-CheckedCommand $Python -m venv $ResolvedVenvDir

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python was not found: $VenvPython"
}

Write-Host "Upgrading packaging tools inside the virtual environment"
Invoke-CheckedCommand $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing minimal build dependencies"
Invoke-CheckedCommand $VenvPython -m pip install -r $BuildRequirements

Write-Host "Building OptCNT with UPX compression using profile '$Profile'"
Invoke-CheckedCommand $VenvPython $BuildScript --profile $Profile

if (-not (Test-Path $ExePath)) {
    throw "Build finished but the executable was not found: $ExePath"
}

$SizeMB = [Math]::Round((Get-Item $ExePath).Length / 1MB, 2)
Write-Host "Build complete: $ExePath"
Write-Host "Executable size: $SizeMB MB"
