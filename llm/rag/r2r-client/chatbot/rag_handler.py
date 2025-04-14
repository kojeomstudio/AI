import os
import json
from r2r import R2RClient
from utils import get_config_path

def send_query(query, mode):

    config_path = get_config_path("config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    client = R2RClient(config["server_url"])

    final_result = None

    if mode == "agent":
        response = client.retrieval.agent(
            message={
                "role": "user",
                "content": f"{query}"
            },
            search_settings={
                "use_semantic_search": True,
                "limit": 2
        })
        try:
            msg = response["results"]["messages"]
            meta_data = msg[0]["metadata"]
            search_result = meta_data["aggregated_search_result"]
            final_result = search_result
        except TypeError:
            final_result = response.results.messages[0].metadata.aggregated_search_result

    elif mode == "rag":
        response = client.retrieval.rag(
            query=query,
            rag_generation_config=config["rag_generation_config"]
        )

        try:
            final_result = response['results']['generated_answer']
        except TypeError:
            final_result = response.results.generated_answer

    return final_result
    
    

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
