# 1. Python 기반 이미지 사용
FROM python:3.12-slim

# 2. 작업 디렉토리 생성 및 설정
WORKDIR /app

# 3. 필요한 파일 복사
COPY core /app/core
COPY start.sh /app
COPY requirements.txt /app

# 4. 환경 변수 설정 (FastAPI와 Uvicorn 서버용)
ENV HOST=0.0.0.0
ENV PORT=9000

CMD ["bash", "start.sh"]