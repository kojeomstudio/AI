import sys, os

def get_path(in_origin: str):
    if getattr(sys, 'frozen', False):  # PyInstaller로 빌드된 실행파일인 경우
        script_dir = os.path.dirname(sys.executable)
    else:  # 일반 Python 스크립트 실행
        script_dir = os.path.dirname(os.path.abspath(__file__))

    target_file_path = os.path.join(script_dir, in_origin)
    return target_file_path
