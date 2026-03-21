# FFmpeg Binary & Build Helper for ClipMaster
$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
$ffmpegSubmoduleDir = Join-Path $scriptRoot "..\FFmpeg"
$binOutputDir = Join-Path $scriptRoot "..\bin\clip-master"

if (!(Test-Path $binOutputDir)) {
    New-Item -ItemType Directory -Path $binOutputDir -Force | Out-Null
}

$ffmpegExe = Join-Path $binOutputDir "ffmpeg.exe"
$ffprobeExe = Join-Path $binOutputDir "ffprobe.exe"

# 1. Check if already exists in bin
if ((Test-Path $ffmpegExe) -and (Test-Path $ffprobeExe)) {
    Write-Host "FFmpeg binaries already present in bin directory." -ForegroundColor Green
    return
}

# 2. Try to copy from submodule (if built manually)
$sourcePaths = @(
    (Join-Path $ffmpegSubmoduleDir "ffmpeg.exe"),
    (Join-Path $ffmpegSubmoduleDir "ffprobe.exe")
)

$foundInSubmodule = $true
foreach ($path in $sourcePaths) {
    if (Test-Path $path) {
        Copy-Item -Path $path -Destination $binOutputDir -Force
        Write-Host "Copied $(Split-Path $path -Leaf) from submodule." -ForegroundColor Green
    } else {
        $foundInSubmodule = $false
    }
}

if ($foundInSubmodule) { return }

# 3. Fallback: Download pre-built binaries to ensure "No Issues"
Write-Host "FFmpeg binaries not found. Downloading pre-built binaries to ensure build success..." -ForegroundColor Yellow
$zipUrl = "https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip"
$tempZip = Join-Path $env:TEMP "ffmpeg_temp.zip"
$tempExtract = Join-Path $env:TEMP "ffmpeg_extract"

if (Test-Path $tempExtract) { Remove-Item -Recurse -Force $tempExtract }

Invoke-WebRequest -Uri $zipUrl -OutFile $tempZip
Expand-Archive -Path $tempZip -DestinationPath $tempExtract

$downloadedFfmpeg = Get-ChildItem -Path $tempExtract -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
$downloadedFfprobe = Get-ChildItem -Path $tempExtract -Filter "ffprobe.exe" -Recurse | Select-Object -First 1

if ($downloadedFfmpeg -and $downloadedFfprobe) {
    Copy-Item -Path $downloadedFfmpeg.FullName -Destination $binOutputDir -Force
    Copy-Item -Path $downloadedFfprobe.FullName -Destination $binOutputDir -Force
    Write-Host "FFmpeg binaries downloaded and deployed successfully." -ForegroundColor Green
} else {
    Write-Error "Failed to locate FFmpeg binaries in downloaded archive."
}

# Cleanup
Remove-Item $tempZip -ErrorAction SilentlyContinue
Remove-Item -Recurse $tempExtract -ErrorAction SilentlyContinue
