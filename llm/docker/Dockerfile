# 베이스 이미지 선택 (Python 3.12 사용)
FROM python:3.12

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 소스 코드 복사
COPY . /app/OpenManus

# 작업 디렉토리 이동
WORKDIR /app/OpenManus

RUN python -m venv my_open_manus_venv && \
    ./my_open_manus_venv/bin/pip install --no-cache-dir -r requirements.txt

RUN chmod +x start.sh

EXPOSE 5172
# 컨테이너 실행 시 start.sh 실행
CMD ["/bin/bash", "-c", "./start.sh"]
