$mp = [Environment]::GetEnvironmentVariable("PATH","Machine").Split(";") | Where-Object { $_ -ne "" }

$userProfile = [Environment]::GetFolderPath("LocalApplicationData")
$remove = @(
  'C:\Program Files (x86)\Python311-32\Scripts\',
  'C:\Program Files (x86)\Python311-32\',
  "$userProfile\Programs\Python\Python37-32\Scripts\",
  "$userProfile\Programs\Python\Python37-32\"
)
$mp2 = $mp | Where-Object { $remove -notcontains $_ }

if (($mp2 -join ';') -ne ($mp -join ';')) {
  [Environment]::SetEnvironmentVariable("PATH", ($mp2 -join ';'), "Machine")
  "Machine PATH 업데이트 완료. 새 콘솔/서비스 재시작 필요."
} else {
  "Machine PATH 변화 없음."
}
