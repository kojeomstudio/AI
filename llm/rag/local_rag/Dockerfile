
# 만약, 도커파일 진행 중에 credential 또는 failed to resolve source metadata for docker.io
# 문제가 발생한다면, ~/.docker/config.json 에서  "credsStore" : "desktop" 항목을 삭제한다.
# -> https://github.com/docker/docker-credential-helpers/issues/149#issuecomment-566832756

FROM python:3.12-slim

WORKDIR /app

COPY ./simple/requirements.txt .
# 필수 패키지 설치를 위한 업데이트
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -r requirements.txt

# WORKDIR은 목적지 경로의 루트 디렉터리.
COPY ./simple ./dist

RUN ./dist/start.sh