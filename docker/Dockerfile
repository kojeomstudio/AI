#
FROM python:3.12

# 작업 디렉토리를 설정
WORKDIR /app

# 필요한 의존성 파일을 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드를 복사
COPY . .

# 컨테이너 시작 시 실행할 명령어 (예: main.py)
#CMD ["python", "main.py"]
