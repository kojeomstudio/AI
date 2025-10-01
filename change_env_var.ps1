# 현재 Machine PATH 가져오기
$mp = [Environment]::GetEnvironmentVariable("PATH","Machine").Split(";") | Where-Object { $_ -ne "" }

# 불필요 경로 필터링 제거
$remove = @(
  'C:\Program Files (x86)\Python311-32\Scripts\',
  'C:\Program Files (x86)\Python311-32\',
  'C:\Users\kojeo\AppData\Local\Programs\Python\Python37-32\Scripts\',
  'C:\Users\kojeo\AppData\Local\Programs\Python\Python37-32\'
)
$mp2 = $mp | Where-Object { $remove -notcontains $_ }

# 변경이 있을 때에만 반영
if (($mp2 -join ';') -ne ($mp -join ';')) {
  [Environment]::SetEnvironmentVariable("PATH", ($mp2 -join ';'), "Machine")
  "Machine PATH 업데이트 완료. 새 콘솔/서비스 재시작 필요."
} else {
  "Machine PATH 변화 없음."
}
