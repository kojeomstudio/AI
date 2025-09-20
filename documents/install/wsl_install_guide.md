※※※※※※※※ WSL2 설치 ※※※※※※※※

(1) 윈도우 기능 활성화. (wsl 가상화 관련)
 - dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
 - dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

 -> 관리자 권한 필요.
 -> dism ?
    - deployment imaging service and management tool : 윈도우 시스템 이미지 도구

(2) 컴퓨터 재부팅 필요.

(3) wsl --install -d Ubuntu-24.04
 - wsl --install --d Ubuntu-24.04 --name lecture-my-ubuntu-24.04
    (특정 이름을 지정해서 설치 가능.)

(4) wsl --set-default-version 2

(5) wsl 
 - 디폴트로 설정된 배포판으로 접속.
 - 특정 배포판으로 선택하려면 -d 옵션을 사용. ( ex: wsl -d my-ubuntu)


※ 기타 명령어.
wsl --list --verbose // 현재 wsl에 설치된 배포판 목록 확인.
wsl -l -v
wsl --status
wsl --unregister <배포판_이름>

 \\wsl.localhost\Ubuntu-24.04\var\lib\docker\volumes
 -> 폴더 정리 후에는 docker 데몬 재시작 필요.
    ( sudo systemctl restart docker )

※ 리눅스 관련.
sudo (super user do)


※※※※※※※※ 도커 설치 (WSL2 - ubuntu 기반) ※※※※※※※※

(1) 기본 설치 관리자 업데이트.
    sudo apt update && sudo apt upgrade -y
     - advanced package tool (apt) : 리눅스 배포판에서 패키지를 관리하는 명령어 도구.


(2) 도커 의존성 설치.
    sudo apt install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

(3) 도커 공식 repo key 추가.
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

(4) 도커 공식 repo 추가.
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

(5) 패키지 업데이트 & docker 설치.
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

(6) systemd 활성화.

sudo tee /etc/wsl.conf <<EOF
[boot]
systemd=true
EOF
 - wsl2 에서는 systemd 기본 비활성화 ( systemd ? 리눅스 기본 서비스 관리자. )

(7) wsl 재실행.
wsl --shutdown (서비스 관리자 변경에 따라 재실행 필수)

(8) docker 서비스 확인.
sudo systemctl daemon-reload
sudo systemctl enable --now docker
sudo systemctl status docker

(9) 현재 사용자를 docker 그룹에 추가.
sudo usermod -aG docker $USER
 - 로그인을 다시 해야함.