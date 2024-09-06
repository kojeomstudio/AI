
# pip install -U langchain
# pip install -U langchain-community
# pip install  pandas
# pip install unstructured
# pip install "unstructured[pdf]"
# pip install -U langgraph
# pip install -U langchain-chroma

# -U 옵션은 upgrade를 의미. / 없으면 최신 버전으로 설치.

import os

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain.chains import RetrievalQA

from langchain.document_loaders import DirectoryLoader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter


ollama_model_name = "EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_S.gguf:latest"
ollama_service_url = 'http://localhost:11434'

# 벡터 스토어를 저장할 경로 설정
vectorstore_db_path = "./chroma_db"

documents_path = "./test_docs"

# .txt, .pdf, .docx 파일 로더
loader = DirectoryLoader(path=documents_path, glob='**/*.txt')
txt_files = loader.load()

total_data = txt_files

print(f"txt files num : {len(txt_files)}")

pdf_loader = DirectoryLoader(path=documents_path, glob='**/*.pdf')
pdf_files = pdf_loader.load()

print(f"pdf_fiels num : {len(pdf_files)}")

total_data += pdf_files


# .doc 및 .docx 파일 로더
docx_loader = DirectoryLoader(path=documents_path, glob='**/*.docx')
docx_files = docx_loader.load()

print(f"docx_files num : {len(docx_files)}")

total_data += docx_files

doc_loader = DirectoryLoader(path=documents_path, glob='**/*.doc')
doc_files = doc_loader.load()  # doc 확장자 파일 추가

print(f"doc_files num : {len(doc_files)}")

total_data += doc_files

# .xlsx 파일 로드
xlsx_files = DirectoryLoader(path=documents_path, glob='**/*.xlsx').load()
for xlsx_file in xlsx_files:
    # pandas를 사용하여 엑셀 파일 읽기
    df = pd.read_excel(xlsx_file.path)
    # 엑셀 데이터를 텍스트로 변환하여 데이터에 추가
    for index, row in df.iterrows():
        text_data = " ".join([str(cell) for cell in row])
        total_data.append({"text": text_data})

# 텍스트를 500자 단위로 나눕니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(total_data)

# 벡터 스토어가 이미 존재하는지 확인
if not os.path.exists(vectorstore_db_path):
    # 벡터 스토어가 없을 때만 문서 임베딩 및 저장
    print("벡터 스토어가 존재하지 않으므로, 새로 생성합니다.")
    
    # 텍스트를 500자 단위로 나눕니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(total_data)
    
    # Ollama 모델을 사용한 임베딩 생성 및 벡터 스토어 구축
    oembed = OllamaEmbeddings(base_url=ollama_service_url, model=ollama_model_name)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory=vectorstore_db_path)
    
    # 벡터 스토어 저장
    vectorstore.persist()
else:
    # 벡터 스토어가 존재할 경우, 저장된 데이터 불러오기
    print("기존 벡터 스토어를 불러옵니다.")
    vectorstore = Chroma(persist_directory=vectorstore_db_path, embedding_function=OllamaEmbeddings(base_url=ollama_service_url, model=ollama_model_name))

# 쿼리를 벡터 스토어에 전달하여 유사 문서를 찾습니다.
question = "중국 위안화 강세 및 추세에 대한 이야기를 해줘."
result_docs = vectorstore.similarity_search(question)
#print(f"docs len : {len(docs)}, docs -> {docs}")

if len(result_docs) == 0:
    print(f"유사한 문서를 찾지 못했습니다.")
else:
    print(f"총 {len(result_docs)}개의 유사 문서를 찾았습니다.")

# Ollama 모델을 사용하여 QA 체인을 만듭니다.
ollama = Ollama(base_url=ollama_service_url, model=ollama_model_name)
qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

try:
    response = qachain.invoke({"query": question})
    print(f"답변: {response['result']}")
except Exception as e:
    print(f"쿼리 처리 중 에러 발생: {e}")

#llm = Ollama(
#    model=ollama_model_name, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
#)
#llm("The first man on the summit of Mount Everest, the highest peak on Earth, was ...")
