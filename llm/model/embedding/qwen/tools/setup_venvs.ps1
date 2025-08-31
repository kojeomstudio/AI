Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Recreates two Python virtual environments at the repo root:
# - .venv      from requirements.txt
# - .venv_gtx  from requirements_gtx.txt
# If the venv directory exists, it is deleted and recreated.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path

$ReqBase = Join-Path $RepoRoot 'requirements.txt'
$ReqGtx  = Join-Path $RepoRoot 'requirements_gtx.txt'

function Assert-FileExists([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "Requirements file not found: $Path"
  }
}

function New-CleanVenv([string]$VenvPath) {
  if (Test-Path -LiteralPath $VenvPath) {
    Write-Host "[setup] Removing existing venv: $VenvPath"
    Remove-Item -LiteralPath $VenvPath -Recurse -Force
  }

  if (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "[setup] Creating venv with 'py -3' at: $VenvPath"
    & py -3 -m venv $VenvPath
  }
  elseif (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "[setup] Creating venv with 'python' at: $VenvPath"
    & python -m venv $VenvPath
  }
  elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    Write-Host "[setup] Creating venv with 'python3' at: $VenvPath"
    & python3 -m venv $VenvPath
  }
  else {
    throw 'Python not found. Please install Python 3.8+ and re-run.'
  }
}

function Install-Requirements([string]$VenvPath, [string]$ReqFile) {
  $pybin = Join-Path $VenvPath 'Scripts/python.exe'
  if (-not (Test-Path -LiteralPath $pybin)) {
    $pybin = Join-Path $VenvPath 'bin/python'
  }

  Write-Host "[setup] Upgrading pip"
  & $pybin -m pip install --upgrade pip | Out-Host

  Write-Host "[setup] Installing dependencies from $(Split-Path -Leaf $ReqFile)"
  & $pybin -m pip install -r $ReqFile | Out-Host

  Write-Host "[setup] Validating installed packages (pip check)"
  try {
    & $pybin -m pip check | Out-Host
  } catch {
    # Non-zero may occur for warnings; continue
    Write-Warning $_
  }

  Write-Host "[setup] Python version in venv:"
  & $pybin -c "import sys; print(sys.version)" | Out-Host
}

Write-Host "[setup] Starting environment setup in: $RepoRoot"

Assert-FileExists -Path $ReqBase
Assert-FileExists -Path $ReqGtx

$VenvBase = Join-Path $RepoRoot '.venv'
$VenvGtx  = Join-Path $RepoRoot '.venv_gtx'

New-CleanVenv -VenvPath $VenvBase
Install-Requirements -VenvPath $VenvBase -ReqFile $ReqBase

New-CleanVenv -VenvPath $VenvGtx
Install-Requirements -VenvPath $VenvGtx -ReqFile $ReqGtx

Write-Host "[setup] All done. Created venvs: $VenvBase and $VenvGtx"

