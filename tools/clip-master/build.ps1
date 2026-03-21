# ClipMaster Build Script
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$appProject = Join-Path $root "ClipMaster.App"
# Binaries should be created in tools/bin/<tool-name>
$parentDir = Split-Path $root -Parent
$outputDir = Join-Path $parentDir "bin\clip-master"

Write-Host "Cleaning up bin directory..." -ForegroundColor Cyan
if (Test-Path $outputDir) {
    Remove-Item -Path $outputDir -Recurse -Force
}
New-Item -Path $outputDir -ItemType Directory | Out-Null

Write-Host "Restoring and Building project..." -ForegroundColor Cyan
dotnet publish $appProject -c Release -o $outputDir --self-contained false

Write-Host "Build Completed. Binaries are in: $outputDir" -ForegroundColor Green

# Copy FFmpeg binaries if they exist in the submodule
$ffmpegScript = Join-Path $root "copy_ffmpeg.ps1"
if (Test-Path $ffmpegScript) {
    Write-Host "Invoking FFmpeg copy script..." -ForegroundColor Cyan
    & $ffmpegScript
}

Write-Host "Note: Ensure 'ffmpeg.exe' and 'ffprobe.exe' are in the PATH or same directory as the executable." -ForegroundColor Yellow
