
# pip install ollama, langchain, langchain_community, chromadb
# (파이썬 가상환경 세팅 후 진행 필요.)
#
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

model_name="EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_S.gguf:latest"

llm = Ollama(
    model=model_name, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
llm("The first man on the summit of Mount Everest, the highest peak on Earth, was ...")