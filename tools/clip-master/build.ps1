# ClipMaster Build Script
$ErrorActionPreference = "Stop"

$root = Get-Location
$appProject = Join-Path $root "ClipMaster.App"
$outputDir = Join-Path $root "bin"

Write-Host "Cleaning up bin directory..." -ForegroundColor Cyan
if (Test-Path $outputDir) {
    Remove-Item -Path $outputDir -Recurse -Force
}
New-Item -Path $outputDir -ItemType Directory | Out-Null

Write-Host "Restoring and Building project..." -ForegroundColor Cyan
dotnet publish $appProject -c Release -o $outputDir --self-contained false

Write-Host "Build Completed. Binaries are in: $outputDir" -ForegroundColor Green
Write-Host "Note: Ensure 'ffmpeg.exe' and 'ffprobe.exe' are in the PATH or same directory as the executable." -ForegroundColor Yellow
