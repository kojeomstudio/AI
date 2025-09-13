# ./docker/ollama/Dockerfile
FROM ollama/ollama:latest

# 네트워크 바인딩 및 모델 디렉터리 경로
ENV OLLAMA_HOST=0.0.0.0 \
    OLLAMA_MODELS=/root/.ollama

# (선택) Modelfile 등 커스텀 모델 레시피를 포함하려면 여기에 복사
# COPY modelfiles/ /opt/ollama/modelfiles/

# 엔트리포인트는 기본값(ollama serve) 사용
# HEALTHCHECK는 compose에서 지정
