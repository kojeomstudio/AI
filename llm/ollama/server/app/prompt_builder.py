from app.config_loader import CONFIG

def build_prompt(user_query: str) -> str:
    """config에 저장된 템플릿을 기반으로 프롬프트 완성"""
    template = CONFIG.get('prompt_template', "{query}")
    return template.replace("{query}", user_query)
