import json, os, pathlib
from app.utils import get_full_path, get_base_path

def load_config():
    """config.json 파일 로드"""
    dir = pathlib.Path(get_base_path()).parent
    config_path = os.path.join(dir, 'config', 'config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()
