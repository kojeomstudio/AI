
import os
import pandas as pd

# 웹서버 동작을 위해 추가.==========================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
#=============================================

from langchain.llms import Ollama
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

from langgraph.graph import StateGraph, END, START

from typing import Annotated
from typing_extensions import TypedDict

from enum import Enum

# pip install --upgrade langchain langsmith langchain-chroma langchain-community pydantic langgraph

# https://pypi.org/project/langgraph/0.0.24/


############################ global variables ####################################
ollama_model_name = "EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_S.gguf:latest"
ollama_service_url = 'http://localhost:11434'

# 벡터 스토어를 저장할 경로 설정
vectorstore_db_path = "./chroma_db"
documents_path = "./test_docs"

# 노드 네임 정의
load_node_name = "load_node"
embedding_node_name = "embedding_node"
search_documents_node_name = "search_doucments_node"
query_node_name = "query_node"

# FastAPI 서버 생성
api_app_server = FastAPI()

DEBUG_LOG_PREFIX_LANGGRAPH = "[LANGGRAPH_DEBUG_LOG]"
DEBUG_LOG_PREFIX_API_APP = "[API_APP_DEBUG_LOG]"

###################################################################################

# 사용자 질문을 받을 모델 정의
class QueryRequest(BaseModel):
    user_query: str 

class StateResultType(Enum):
    NONE = 0
    SUCCESS = 1
    FAIL_NONE_DOCUMENTS = 2
    FAIL_NOT_FOUND_DOCUMENTS = 3
    FAIL_QUERY_ERROR = 4
    FAIL_MAKE_VECTORSTORE = 5
    FAIL_UNKNOWN = 100

class StateTypeDict(TypedDict):
    documents : list
    similarity_documents : list
    vectorstore : Chroma
    user_query : str
    llm_answer : str
    state_result_msg : str
    state_result_type : StateResultType

class ErrorHandler:
    @staticmethod
    def check(state : StateTypeDict):
        return True
    
class Embedding_Helper:
    @staticmethod
    def MakeVectorStoreDB(documents_source : list):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents_source)

        oembed = OllamaEmbeddings(base_url=ollama_service_url, model=ollama_model_name)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory=vectorstore_db_path)
        #vectorstore.persist()

        return f"기존에 존재하는 db가 없으므로 신규 생성합니다."

class LogType(Enum):
    LANGGRAPH = 0
    API_APP = 1

class SimpleLogger:
    @staticmethod
    def Log(msg : str, log_type : LogType):

        prefix = ""
        if log_type == LogType.LANGGRAPH:
            prefix = DEBUG_LOG_PREFIX_LANGGRAPH
        elif log_type == LogType.API_APP:
            prefix = DEBUG_LOG_PREFIX_API_APP
            
        print(f"{prefix} {msg}")

# 문서 로딩 노드
def state_load_documents(state : StateTypeDict):
    loader = DirectoryLoader(path=documents_path, glob='**/*.txt')
    txt_files = loader.load()

    total_data = txt_files
    #print(f"txt files num : {len(txt_files)}")

    pdf_loader = DirectoryLoader(path=documents_path, glob='**/*.pdf')
    pdf_files = pdf_loader.load()
    #print(f"pdf files num : {len(pdf_files)}")
    total_data += pdf_files

    docx_loader = DirectoryLoader(path=documents_path, glob='**/*.docx')
    docx_files = docx_loader.load()
    #print(f"docx_files num : {len(docx_files)}")
    total_data += docx_files

    doc_loader = DirectoryLoader(path=documents_path, glob='**/*.doc')
    doc_files = doc_loader.load()
    #print(f"doc_files num : {len(doc_files)}")
    total_data += doc_files

    xlsx_files = DirectoryLoader(path=documents_path, glob='**/*.xlsx').load()
    for xlsx_file in xlsx_files:
        df = pd.read_excel(xlsx_file.path)
        for index, row in df.iterrows():
            text_data = " ".join([str(cell) for cell in row])
            total_data.append({"text": text_data})

    file_type_string = f"txt : {len(txt_files)}, pdf : {len(pdf_files)}, docx : {len(docx_files)}, xlsx : {len(xlsx_files)}"

    result_msg = ""
    result_type = StateResultType.NONE
    if len(total_data) > 0:
        result_type = StateResultType.SUCCESS
        result_msg = f"로딩에 성공했습니다. files : {file_type_string}"
    else:
        result_type = StateResultType.FAIL_NONE_DOCUMENTS
        result_msg = f"로딩에 성공한 문서가 없습니다."

    state['documents'] = total_data
    state['state_result_type'] = result_type
    state['state_result_msg'] = result_msg

    return state


# 벡터 스토어 처리 노드 ( 임베딩 )
def state_embedding(state : StateTypeDict):

    documents_source = state['documents']

    vectorstore_process_msg = ""

    if not os.path.exists(vectorstore_db_path):
        vectorstore_process_msg = Embedding_Helper.MakeVectorStoreDB(documents_source)
    else:
        vectorstore = Chroma(persist_directory=vectorstore_db_path, embedding_function=OllamaEmbeddings(base_url=ollama_service_url, model=ollama_model_name))

        collected_docs = []
        for index in range(len(vectorstore.get()["ids"])):
            doc_metadata = vectorstore.get()["metadatas"][index]
            collected_docs.append(doc_metadata["source"])

        new_docs = []
        for data in documents_source:
            new_docs.append(data.metadata['source'])

        is_new_docs_subset = set(new_docs).issubset(set(collected_docs))

        if not is_new_docs_subset:
            vectorstore_process_msg = Embedding_Helper.MakeVectorStoreDB(documents_source)
        else:
            vectorstore_process_msg = f"신규 문서가 없으므로, 기존 벡터스토어 DB를 사용합니다."

    result_msg = ""
    result_type = StateResultType.NONE
    if vectorstore != None:
        result_type = StateResultType.SUCCESS
        result_msg = f"임베딩에 성공했습니다. detail : {vectorstore_process_msg}"
    else:
        result_type = StateResultType.FAIL_MAKE_VECTORSTORE
        result_msg = f"임베딩에 실패했습니다. detail : {vectorstore_process_msg}"

    state['vectorstore'] = vectorstore
    state['state_result_type'] = result_type
    state['state_result_msg'] = result_msg

    return state

# 유사 문서 검색 노드
def state_search_similarity_documents(state: StateTypeDict):

    result_type = StateResultType.NONE

    vectorstore = state['vectorstore']
    question = state['user_query']

    result_docs = vectorstore.similarity_search(question)

    result_msg = ""

    if len(result_docs) == 0:
        result_type = StateResultType.FAIL_NOT_FOUND_DOCUMENTS
        result_msg = f"유사한 문서를 찾기 못했습니다."
    else:
        result_type = StateResultType.SUCCESS
        result_msg = f"모두 {len(result_docs)}개의 유사 문서를 찾았습니다."

    state['similarity_documents'] = result_docs
    state['state_result_type'] = result_type
    state['state_result_msg'] = result_msg

    return state

def state_query(state : StateTypeDict):

    vectorstore = state['vectorstore']
    question = state['user_query']
    
    ollama = Ollama(base_url=ollama_service_url, model=ollama_model_name)
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

    result_msg = ""
    result_type = StateResultType.NONE

    try:
        response = qachain.invoke({"query": question}) # response['result']
        result_type = StateResultType.SUCCESS
    except Exception as e:
        result_type = StateResultType.FAIL_QUERY_ERROR
        result_msg = f"쿼리 처리 중 에러 발생: {e}"

    if (response != None) and (response['result']):
        state['llm_answer'] = response['result']
    else:
        state['llm_answer'] = f"답변을 할 수 없습니다."

    state['state_result_type'] = result_type
    state['state_result_msg'] = result_msg

    return state

# 상태 그래프를 사용하여 노드 구성 및 실행
def build_stategraph():
    
    # 그래프 생성
    graph = StateGraph(StateTypeDict)

    # 노드 추가
    graph.add_node(load_node_name, state_load_documents)
    graph.add_node(embedding_node_name, state_embedding)
    graph.add_node(search_documents_node_name, state_search_similarity_documents)
    graph.add_node(query_node_name, state_query)

    # 노드 연결
    # ( START -> load -> embedding -> search(documents) -> query -> END 흐름으로 처리)
    graph.add_edge(START, load_node_name)
    graph.add_edge(load_node_name, embedding_node_name)
    graph.add_edge(embedding_node_name, search_documents_node_name)
    graph.add_edge(search_documents_node_name, query_node_name)
    graph.add_edge(query_node_name, END)

    # 컴파일. ( 그래프에 설정된 노드, 엣지, 컨디셔널등을 컴파일한다. )
    app = graph.compile()
    return app

def process_user_query(in_user_query : str):
    # build graph
    app_result = build_stategraph()

    # 유저 쿼리만 필요하므로 해당 내용만 설정.
    inputs = {'user_query' : in_user_query}

    for output in app_result.stream(inputs):
        # 출력된 결과에서 키와 값을 순회합니다.
        for key, value in output.items():

            SimpleLogger.Log(f"==================================", LogType.LANGGRAPH)
            SimpleLogger.Log(f"Output from node name : {key}", LogType.LANGGRAPH)
            
            message = value['state_result_msg']
            type = value['state_result_type']

            SimpleLogger.Log(f"state result msg : {message} type : {type}", LogType.LANGGRAPH)

            # 쿼리 노드인 경우, 유저 질의와 llm의 대답을 출력한다.
            # 해당 내용이 empty 인 경우는 없을 것
            if key == query_node_name:
                result_llm_answer = value.get('llm_answer')
                result_user_query = value.get('user_query')

                SimpleLogger.Log(f"user_query : {result_user_query}", LogType.LANGGRAPH)
                SimpleLogger.Log(f"llm_answer : {result_llm_answer}", LogType.LANGGRAPH)

                return result_llm_answer


# API 엔드포인트 정의
@api_app_server.post("/query")
async def query_api_process(query: QueryRequest):
    SimpleLogger.Log(f"query_api : user_query : {query.user_query}", LogType.API_APP)
    try:
        llm_answer = process_user_query(query.user_query)
        SimpleLogger.Log(f"llm(+rag) answer : {llm_answer}", LogType.API_APP)
        return {"llm_answer": llm_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app_server.get("/")
async def root_api_process():
    SimpleLogger.Log(f"rag app server root! ", LogType.API_APP)