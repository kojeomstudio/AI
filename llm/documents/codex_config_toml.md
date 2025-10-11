# 사용자 디렉터리 >> .codex/config.toml

model = "gpt-5-codex"
model_reasoning_effort = "high"
[projects."/Users/kojeomstudio/HELLO_MY_WORLD"]
trust_level = "trusted"


# codex-cli의 경우, provider와 모델을 선언해놓고 프로파일로 로드해서 사용할 수 있다.
# gpt 모델만 사용가능할 것 같지만 실제로 로컬 오픈 모델도 사용 가능한 구조.

[model_providers.ollama]
name = "Ollama"
base_url = "http://localhost:11434/v1"

#[profiles.gpt-oss-120b-ollama]
#model_provider = "ollama"
#model = "gpt-oss:120b"

[profiles.qwen3-1p7b-ollama]
model_provider = "ollama"
model = "qwen3:1.7b"

[model_providers.lm_studio]
name = "LM Studio"
base_url = "http://localhost:1234/v1"

[profiles.qwen3-coder-4b-lms]
model_provider = "lm_studio"
model = "qwen/qwen3:4b"