
# (파이썬 가상환경 세팅 후 진행 필요.)
# pip install 
# -> ollama
# -> chromadb
# -> sentence-transformers
# -> llama-index-vector-stores-chroma
# -> 
#

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PyMuPDFReader,
)
from llama_index import VectorStoreIndex
from llama_index import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores import ChromaVectorStore
from langchain.embeddings import SentenceTransformerEmbeddings
import chromadb
import ollama

ollama_model_name="EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_S.gguf:latest"

# 특정 폴더 경로 지정
folder_path = './test_docs'

# 문서 읽기 (PDF, DOCX, TXT 등 지원)
documents = SimpleDirectoryReader(folder_path).load_data()

# Settings 클래스를 사용하여 기본 설정 정의
settings = Settings(
    embed_model=LangchainEmbedding(SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")),
    chunk_size=512,
    max_input_size=4096
)

# Chroma Vector Store 초기화
chroma_client = chromadb.Client()
vector_store = chromadb.vectorstores.ChromaVectorStore(client=chroma_client)

# VectorStoreIndex 생성 및 문서 추가
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    settings=settings
)

# 쿼리 전처리 및 벡터 검색
query = "금일 주식 시장에 대한 평가는 어떻게 될까?"
query_engine = index.as_query_engine()
preprocessed_query = query_engine.query(query)

# ollama API를 사용하여 LLM에 질의 전달 및 결과 출력
response = ollama.generate(
    model=ollama_model_name,  # Ollama 모델 이름
    prompt=str(preprocessed_query)  # 전처리된 질의를 prompt로 사용
)

# 결과 출력
print(response['text'])