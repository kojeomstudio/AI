
# pip install langchain
# pip install -U langchain-community
# pip install pandas
# pip install unstructured
# pip install "unstructured[pdf]"


from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA

from langchain.document_loaders import DirectoryLoader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter


ollama_model_name = "EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_S.gguf:latest"
ollama_service_url = 'http://localhost:11434'

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


docx_loader = DirectoryLoader(path=documents_path, glob='**/*.docx')
docx_files = docx_loader.load()

print(f"docx_files num : {len(docx_files)}")

total_data += docx_files

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

# Ollama 모델을 사용한 임베딩 생성 및 벡터 스토어 구축
oembed = OllamaEmbeddings(base_url=ollama_service_url, model=ollama_model_name)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

# 쿼리를 벡터 스토어에 전달하여 유사 문서를 찾습니다.
question = "위완화 강세는 언제까지 이어질까? 해당 내용이 궁금해"
docs = vectorstore.similarity_search(question)
#print(f"docs len : {len(docs)}, docs -> {docs}")

# Ollama 모델을 사용하여 QA 체인을 만듭니다.
ollama = Ollama(base_url=ollama_service_url, model=ollama_model_name)
qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
qachain.invoke({"query": question})

#llm = Ollama(
#    model=ollama_model_name, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
#)
#llm("The first man on the summit of Mount Everest, the highest peak on Earth, was ...")

