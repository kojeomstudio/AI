#requires -Version 5.1
<#
  mabinogi-2d-like 통합 개발 실행 스크립트.
  서버(ASP.NET Core) + Godot 클라이언트 N개를 한 번에 띄워 멀티플레이를 테스트한다.

  사용 예:
    ./run-dev.ps1                      # 서버 + 클라 2개(hero1/hero2)
    ./run-dev.ps1 -Clients 3
    ./run-dev.ps1 -NoBuild             # 빌드 생략(이미 빌드됨)
    ./run-dev.ps1 -Godot "D:\Godot\Godot.exe"

  종료: 클라이언트 창을 모두 닫으면 서버가 자동 정리된다. (Ctrl+C 정리는 best-effort)
#>
param(
  [int]$Clients = 2,
  [int]$Port = 5080,
  [string]$Godot = $(if ($env:GODOT_BIN) { $env:GODOT_BIN } else { "C:\workspaces\Godot_v4.6.3-stable_mono_win64\Godot_v4.6.3-stable_mono_win64.exe" }),
  [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$root       = $PSScriptRoot
$serverProj = Join-Path $root "server\Mabinogi2D.Server\Mabinogi2D.Server.csproj"
$serverDll  = Join-Path $root "server\Mabinogi2D.Server\bin\Debug\net8.0\Mabinogi2D.Server.dll"
$clientProj = Join-Path $root "client\Mabinogi2D.Client.csproj"
$clientDir  = Join-Path $root "client"
$baseUrl    = "http://localhost:$Port"

if (-not (Test-Path $Godot)) {
  throw "Godot 실행 파일을 찾을 수 없습니다: $Godot  (-Godot 인자 또는 GODOT_BIN 환경변수로 지정)"
}

# 1) 빌드 (클라이언트는 게임 모드 실행 전에 C# 어셈블리가 필요)
if (-not $NoBuild) {
  Write-Host "[build] 서버..." -ForegroundColor Cyan
  dotnet build $serverProj -c Debug | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "서버 빌드 실패" }
  Write-Host "[build] 클라이언트..." -ForegroundColor Cyan
  dotnet build $clientProj -c Debug | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "클라이언트 빌드 실패" }
}
if (-not (Test-Path $serverDll)) { throw "서버 DLL이 없습니다(빌드 필요): $serverDll" }

$server = $null
$clientProcs = @()
try {
  # 2) 서버 기동 — 빌드된 DLL 직접 실행(단일 프로세스, 포트 결정적, dev 환경=JWT 임시키 자동)
  Write-Host "[server] 기동 @ $baseUrl" -ForegroundColor Green
  $server = Start-Process -FilePath "dotnet" `
    -ArgumentList @("`"$serverDll`"", "--urls", $baseUrl, "--environment", "Development") `
    -PassThru -WindowStyle Minimized

  # 3) 서버 LISTEN 대기 (최대 ~15초)
  $ready = $false
  for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-WebRequest -Uri "$baseUrl/" -UseBasicParsing -TimeoutSec 2 | Out-Null; $ready = $true; break }
    catch { Start-Sleep -Milliseconds 500 }
  }
  if (-not $ready) { throw "서버가 $baseUrl 에서 응답하지 않습니다." }
  Write-Host "[server] 준비 완료 (pid $($server.Id))" -ForegroundColor Green

  # 4) 클라이언트 N개 — 각자 다른 캐릭터로 입장
  for ($n = 1; $n -le $Clients; $n++) {
    $user = "hero$n"; $char = "Hero$n"
    Write-Host "[client $n] $user / $char" -ForegroundColor Yellow
    $p = Start-Process -FilePath $Godot `
      -ArgumentList @("--path", "`"$clientDir`"", "--", "--user", $user, "--char", $char, "--server", $baseUrl) `
      -PassThru
    $clientProcs += $p
    Start-Sleep -Milliseconds 400
  }

  Write-Host "`n실행 중. 화살표키로 이동. 클라이언트 창을 모두 닫으면 서버가 정리됩니다." -ForegroundColor Cyan
  # 5) 모든 클라이언트 종료까지 대기
  foreach ($p in $clientProcs) { $p.WaitForExit() }
}
finally {
  Write-Host "`n[cleanup] 종료 정리..." -ForegroundColor Magenta
  if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue }
  foreach ($p in $clientProcs) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
}
