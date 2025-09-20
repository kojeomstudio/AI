※※※※※※※※ Ollama 설치 ※※※※※※※※

(a) 도커 설치
 -> https://hub.docker.com/r/ollama/ollama (도커허브 공식 이미지)
 -> docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama (CPU only)
 -> docekr exec -it [container-Id] /bin/bash
    (실행중인 컨테이너로 접속하는 명령어 / -i 표준입력 상호작용 가능 / -t 터미널 모드)
 -> ollama 컨테이너로 접속 후 정상 동작 확인 및 모델 pull 필요.
    - ollama pull qwen2.5:0.5b
    - ollama pull mxbai-embed-large

(b) 일반 설치
 -> curl -fsSL https://ollama.com/install.sh | sh

 (GPU 이용) NVIDIA - CUDA 설치
 -> 도커 허브 공식 이미지에 세부 절차 있음.
   ( https://hub.docker.com/r/ollama/ollama )

 (GPU 이용) AMD
 -> docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm

※※※※※※※※ R2R 설치 ※※※※※※※※