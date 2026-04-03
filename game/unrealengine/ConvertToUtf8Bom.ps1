# Source/UnrealWorld 하위 모든 .h/.cpp 파일을 UTF-8 BOM으로 변환
# 실행: powershell -ExecutionPolicy Bypass -File GameTools/ConvertToUtf8Bom.ps1

$sourceDir = "$PSScriptRoot\..\Source\UnrealWorld"
$extensions = @("*.h", "*.cpp", "*.inl")
$encoding = New-Object System.Text.UTF8Encoding $true  # $true = BOM 포함

$files = $extensions | ForEach-Object { Get-ChildItem -Path $sourceDir -Filter $_ -Recurse }
$count = 0

foreach ($file in $files) {
    $rawBytes = [System.IO.File]::ReadAllBytes($file.FullName)

    # 이미 UTF-8 BOM인 경우 (EF BB BF) 스킵
    if ($rawBytes.Length -ge 3 -and $rawBytes[0] -eq 0xEF -and $rawBytes[1] -eq 0xBB -and $rawBytes[2] -eq 0xBF) {
        continue
    }

    # 현재 인코딩으로 텍스트 읽기 후 UTF-8 BOM으로 재저장
    try {
        # CP949(EUC-KR)로 시도, 실패 시 UTF-8 무BOM으로 처리
        $cp949 = [System.Text.Encoding]::GetEncoding(949)
        $text = $cp949.GetString($rawBytes)
        [System.IO.File]::WriteAllText($file.FullName, $text, $encoding)
        Write-Host "Converted (CP949): $($file.FullName)"
        $count++
    }
    catch {
        Write-Warning "Failed: $($file.FullName) — $_"
    }
}

Write-Host "`n완료: $count 개 파일 변환됨"
