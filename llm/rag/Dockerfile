
# 만약, 도커파일 진행 중에 credential 또는 failed to resolve source metadata for docker.io
# 문제가 발생한다면, ~/.docker/config.json 에서  "credsStore" : "desktop" 항목을 삭제한다.
# -> https://github.com/docker/docker-credential-helpers/issues/149#issuecomment-566832756

FROM python:3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# WORKDIR은 목적지 경로의 루트 디렉터리.
COPY ./local_llm_rag_web_server.py .
COPY ./rag_config.json .
COPY ./templates ./templates

CMD ["python", "local_llm_rag_web_server.py"]
