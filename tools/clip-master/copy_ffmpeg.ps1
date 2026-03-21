# FFmpeg Binary Helper for ClipMaster
$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
$ffmpegSubmoduleDir = Join-Path $scriptRoot "..\FFmpeg"
$binOutputDir = Join-Path $scriptRoot "..\bin\clip-master"

# Potential locations for built binaries in the submodule
$sourcePaths = @(
    (Join-Path $ffmpegSubmoduleDir "ffmpeg.exe"),
    (Join-Path $ffmpegSubmoduleDir "ffprobe.exe")
)

Write-Host "Checking for FFmpeg binaries in submodule..." -ForegroundColor Cyan

$foundAll = $true
foreach ($path in $sourcePaths) {
    if (Test-Path $path) {
        $fileName = Split-Path $path -Leaf
        $target = Join-Path $binOutputDir $fileName
        Copy-Item -Path $path -Destination $target -Force
        Write-Host "Copied $fileName to $binOutputDir" -ForegroundColor Green
    } else {
        Write-Host "Missing: $path" -ForegroundColor Yellow
        $foundAll = $false
    }
}

if (-not $foundAll) {
    Write-Host "`nNote: FFmpeg must be built first inside tools/FFmpeg." -ForegroundColor Red
    Write-Host "On Windows, this usually requires MSYS2 or MinGW." -ForegroundColor Red
    Write-Host "Alternatively, manually place 'ffmpeg.exe' and 'ffprobe.exe' into '$binOutputDir'." -ForegroundColor Yellow
}
