# kojeomstudio
# 유틸 함수 모음.

import os, sys

def get_path(in_origin: str):
    # 절대 경로로 들어오면 그대로 반환
    if os.path.isabs(in_origin):
        return in_origin

    # 실행 위치 기준 루트 설정
    if getattr(sys, 'frozen', False):
        script_dir = os.path.dirname(sys.executable)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

    return os.path.normpath(os.path.join(script_dir, in_origin))
