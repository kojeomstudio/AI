# YoutubeExtractor Build Script
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$appProject = Join-Path $root "YoutubeExtractor.App"
# Binaries should be created in Bins/<tool-name>
$repoRoot = (Split-Path $root -Parent).Parent
$outputDir = Join-Path $repoRoot "Bins\youtube-extractor"

Write-Host "Cleaning up bin directory..." -ForegroundColor Cyan
if (Test-Path $outputDir) {
    Remove-Item -Path $outputDir -Recurse -Force
}
New-Item -Path $outputDir -ItemType Directory | Out-Null

Write-Host "Restoring and Building project..." -ForegroundColor Cyan
dotnet publish $appProject -c Release -o $outputDir --self-contained false

Write-Host "Build Completed. Binaries are in: $outputDir" -ForegroundColor Green
