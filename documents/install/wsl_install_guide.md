
# WSL2 및 Docker, Ollama 설치 가이드

## 1. WSL2 설치
**1) 윈도우 기능 활성화 (관리자 권한 필요)**
윈도우 시스템 이미지 도구(dism)를 사용하여 WSL 가상화 관련 기능을 활성화합니다.
```cmd
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
**2) 컴퓨터 재부팅**

**3) Ubuntu 설치 및 설정**
기본적으로 Ubuntu-24.04를 설치하며, 필요시 특정 이름을 지정할 수 있습니다.
```bash
wsl --install -d Ubuntu-24.04
# 특정 이름 지정 설치 예시: wsl --install --d Ubuntu-24.04 --name lecture-my-ubuntu-24.04
```
WSL 기본 버전을 2로 설정합니다.
```cmd
wsl --set-default-version 2
```

**4) WSL 접속 및 주요 명령어**
*   **기본 배포판 접속:** `wsl`
*   **특정 배포판 접속:** `wsl -d <배포판_이름>` (예: `wsl -d my-ubuntu`)
*   **설치된 배포판 목록 확인:** `wsl --list --verbose` 또는 `wsl -l -v`
*   **WSL 상태 확인:** `wsl --status`
*   **배포판 등록 해제:** `wsl --unregister <배포판_이름>`

---

## 2. 도커 (Docker) 설치 (WSL2 환경)

**1) 기본 설치 관리자 업데이트**
```bash
sudo apt update && sudo apt upgrade -y
```

**2) 도커 의존성 설치**
```bash
sudo apt install -y \
ca-certificates \
curl \
gnupg \
lsb-release
```

**3) 도커 공식 Repo Key 추가**
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

**4) 도커 공식 Repo 추가**
```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

**5) 패키지 업데이트 및 도커 설치**
```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**6) systemd 활성화 및 WSL 재실행**
WSL2에서는 리눅스 기본 서비스 관리자인 systemd가 기본적으로 비활성화되어 있으므로 이를 활성화합니다.
```bash
sudo tee /etc/wsl.conf <<EOF
[boot]
systemd=true
EOF
```
설정 후 서비스 관리자 변경 사항을 적용하기 위해 윈도우 환경에서 WSL을 완전히 종료 후 재실행해야 합니다.
```cmd
wsl --shutdown
```

**7) Docker 서비스 확인 및 시작**
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now docker
sudo systemctl status docker
```

**8) 권한 설정**
현재 사용자를 docker 그룹에 추가합니다 (설정 후 로그인을 다시 해야 적용됩니다).
```bash
sudo usermod -aG docker $USER
```
*(참고: 폴더 정리 후에는 `sudo systemctl restart docker` 명령어로 docker 데몬 재시작이 필요할 수 있습니다)*.

---

## 3. Ollama 설치

### (A) 도커(Docker)를 이용한 설치 (공식 이미지 사용)
**1) 컨테이너 실행 (CPU 전용)**
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

**2) 컨테이너 접속 및 모델 다운로드**
실행 중인 컨테이너에 상호작용 가능한 터미널 모드로 접속합니다.
```bash
docker exec -it [container-Id] /bin/bash
```
접속 후 정상 동작 확인 및 사용할 모델을 Pull 받습니다.
```bash
ollama pull qwen2.5:0.5b
ollama pull mxbai-embed-large
```

**3) GPU를 이용한 설치**
*   **NVIDIA - CUDA:** 도커 허브 공식 이미지(https://hub.docker.com/r/ollama/ollama)의 세부 절차를 참고합니다.
*   **AMD:**
```bash
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```

### (B) 일반 스크립트 설치
```bash
curl -fsSL https://ollama.com/install.sh | sh
```