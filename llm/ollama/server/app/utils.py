import os
import sys

def get_base_path():
    """실행 환경에 따라 base 경로 반환 (PyInstaller 대응)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_full_path(relative_path: str) -> str:
    """상대 경로를 절대 경로로 변환."""
    return os.path.join(get_base_path(), relative_path)
