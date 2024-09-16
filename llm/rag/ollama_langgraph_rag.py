
import os
import pandas as pd

from langchain.llms import Ollama
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

from langgraph.graph import StateGraph, END, START

from typing import Annotated
from typing_extensions import TypedDict

# pip install --upgrade langchain langsmith langchain-chroma langchain-community pydantic langgraph

# https://pypi.org/project/langgraph/0.0.24/


################ global variables #####################
ollama_model_name = "EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_S.gguf:latest"
ollama_service_url = 'http://localhost:11434'

# 벡터 스토어를 저장할 경로 설정
vectorstore_db_path = "./chroma_db"
documents_path = "./test_docs"

# 노드 네임 정의
load_node_name = "load_node"
embedding_node_name = "embedding_node"
query_node_name = "query_node"

#####################################################

class StateTypeDict(TypedDict):
    documents : list
    vectorstore : Chroma
    user_query : str

# 1. 문서 로딩 노드
def process_load_documents(state : StateTypeDict):
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

    doc_loader = DirectoryLoader(path=documents_path, glob='**/*.doc')
    doc_files = doc_loader.load()
    print(f"doc_files num : {len(doc_files)}")
    total_data += doc_files

    xlsx_files = DirectoryLoader(path=documents_path, glob='**/*.xlsx').load()
    for xlsx_file in xlsx_files:
        df = pd.read_excel(xlsx_file.path)
        for index, row in df.iterrows():
            text_data = " ".join([str(cell) for cell in row])
            total_data.append({"text": text_data})

    return {"documents" : total_data}


# 2. 벡터 스토어 처리 노드 ( 임베딩 )
def process_embedding(state : StateTypeDict):
    total_data = state['documents']

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(total_data)

    if not os.path.exists(vectorstore_db_path):
        print("벡터 스토어가 존재하지 않으므로, 새로 생성합니다.")
        oembed = OllamaEmbeddings(base_url=ollama_service_url, model=ollama_model_name)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory=vectorstore_db_path)
        vectorstore.persist()
    else:
        print("기존 벡터 스토어를 불러옵니다.")
        vectorstore = Chroma(persist_directory=vectorstore_db_path, embedding_function=OllamaEmbeddings(base_url=ollama_service_url, model=ollama_model_name))

    return {"vectorstore" : vectorstore}


# 3. 쿼리 처리 노드
def process_query(state: StateTypeDict):

    vectorstore = state['vectorstore']
    question = state['user_query']

    result_docs = vectorstore.similarity_search(question)

    if len(result_docs) == 0:
        print(f"유사한 문서를 찾지 못했습니다.")
    else:
        print(f"총 {len(result_docs)}개의 유사 문서를 찾았습니다.")

    ollama = Ollama(base_url=ollama_service_url, model=ollama_model_name)
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

    try:
        response = qachain.invoke({"query": question})
        print(f"답변: {response['result']}")
    except Exception as e:
        print(f"쿼리 처리 중 에러 발생: {e}")

    return {}


# 4. 상태 그래프를 사용하여 노드 구성 및 실행
def build_stategraph():
    
    # 그래프 생성
    graph = StateGraph(StateTypeDict)

    # 노드 추가
    graph.add_node(load_node_name, process_load_documents)
    graph.add_node(embedding_node_name, process_embedding)
    graph.add_node(query_node_name, process_query)

    # 노드 연결
    # ( START -> load -> embedding -> query -> END 흐름으로 처리)
    graph.add_edge(START, load_node_name)
    graph.add_edge(load_node_name, embedding_node_name)
    graph.add_edge(embedding_node_name, query_node_name)
    graph.add_edge(query_node_name, END)


    # 컴파일. ( 그래프에 설정된 노드, 엣지, 컨디셔널등을 컴파일한다. )
    app = graph.compile()
    return app


# build graph
app_result = build_stategraph()

inputs = {'user_query' : "중국 위안화 강세와 추세는 어때?", "documents" : {}, "vectorstore" : None}

for output in app_result.stream(inputs):
    # 출력된 결과에서 키와 값을 순회합니다.
    for key, value in output.items():
        # 노드의 이름과 해당 노드에서 나온 출력을 출력합니다.
        print(f"Output from node '{key}':")
        print("---")
        # 출력 값을 예쁘게 출력합니다.
        print(f" value : {value}")
    # 각 출력 사이에 구분선을 추가합니다.
    print("\n---\n")