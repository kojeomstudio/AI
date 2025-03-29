import sys
import os

def get_config_path(filename):
    if getattr(sys, 'frozen', False):  # PyInstaller 실행
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, filename)
