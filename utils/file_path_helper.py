import os
import sys

def convert_file_path(in_origin : str):
    if getattr(sys, 'frozen', False):  # 바이너리로 실행되었다면
        script_dir = sys._MEIPASS
    else:  # 일반 스크립트 실행
        script_dir = os.path.dirname(os.path.abspath(__file__))

    converted_path = os.path.join(script_dir, in_origin)
    return converted_path