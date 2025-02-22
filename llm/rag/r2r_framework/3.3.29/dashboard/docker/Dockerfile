FROM ubuntu:24.04
LABEL version="1"

WORKDIR /app

COPY ./start.sh /app

SHELL ["/bin/bash", "-c"]

# 최신 Node.js (22.x) 및 PNPM 설치 + 프로젝트 빌드 (단일 RUN 블록)
RUN chmod +x /app/start.sh && \
    apt-get update && \
    apt-get install -y curl git && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g pnpm && \
    git clone https://github.com/SciPhi-AI/R2R-Application.git && \
    cd R2R-Application && \
    pnpm install && \
    pnpm build

EXPOSE 3000

CMD ["/app/start.sh"]