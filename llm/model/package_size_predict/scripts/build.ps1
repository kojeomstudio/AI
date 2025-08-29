Param(
    [string]$VenvPath = ".venv",
    [string]$ReqFile = "requirements.txt",
    [string]$Entry = "run_training.py",
    [string]$Name = "package_size_predict"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (!(Test-Path $VenvPath)) {
  python -m venv $VenvPath
}

$venvPython = Join-Path $VenvPath "Scripts/python.exe"
$venvPip = Join-Path $VenvPath "Scripts/pip.exe"

& $venvPython -m pip install --upgrade pip
& $venvPip install -r $ReqFile

# Ensure PyInstaller is available
& $venvPython -m PyInstaller --onefile --name $Name `
  --hidden-import openpyxl `
  --hidden-import pandas `
  --collect-all pandas `
  --collect-all openpyxl `
  $Entry

Write-Host "Build complete. Binary at .\dist\$Name.exe"

