import os
import json
from r2r import R2RClient
from utils import get_config_path

def send_query(query):
    config_path = get_config_path("config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    client = R2RClient(config["server_url"])
    response = client.retrieval.rag(
        query=query,
        rag_generation_config=config["rag_generation_config"]
    )

    try:
        return response['results']['generated_answer']
    except TypeError:
        return response.results.generated_answer



def get_font_config():
    config_path = get_config_path("config.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        font_conf = config.get("font", {})
        font_family = font_conf.get("family", "맑은 고딕")
        font_size = font_conf.get("size", 11)
        return font_family, font_size
    except Exception:
        return "맑은 고딕", 11
