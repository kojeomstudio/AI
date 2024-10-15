
import os
import pandas as pd
import asyncio
import time

# 웹서버 동작을 위해 추가.==========================
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse  # responses 모듈에서 HTMLResponse 가져오기
from pydantic import BaseModel
#=============================================

#import tiktoken # 입력 토큰화에 사용되는 라이브러리.

from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate # 프롬프트 엔지니어링을 위한 라이브러리.

from langgraph.graph import StateGraph, END, START

from typing import Annotated
from typing_extensions import TypedDict

from enum import Enum

# pip install --upgrade langchain langsmith langchain-chroma langchain-community pydantic langgraph

# https://pypi.org/project/langgraph/0.0.24/

class PerformanceHelper:
    start_time : float

    def start_timer(self):
        self.start_time = time.perf_counter()
    
    def end_timer(self):
        end_time = time.perf_counter()
        execution_time_ms  = ((end_time - self.start_time) * 1000) # to mille second
        execution_time_sec = execution_time_ms * 1000

        return execution_time_ms, execution_time_sec

############################ global variables ####################################
ollama_model_name = "EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_S.gguf:latest"
ollama_service_url = 'http://localhost:11434'

# 벡터 스토어를 저장할 경로 설정
vectorstore_db_path = "./chroma_db"
documents_path = "./test_docs"

# 노드 네임 정의
load_node_name = "load_node"
embedding_node_name = "embedding_node"
search_similarity_documents_node_name = "search_similarity_doucments_node"
query_node_name = "query_node"
make_prompt_template_node_name = "make_prompt_template_node"

# FastAPI 서버 생성
api_app_server = FastAPI()

performance_helper = PerformanceHelper()

default_time_out_sec = 120

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
    FAIL_MAKE_PROMPT_TEMPLATE = 6
    FAIL_UNKNOWN = 100

class GraphQueryType(Enum):
    NONE = 0 # error
    USER_REQUEST = 1 # 유저 쿼리
    DOCS_EMBEDDING = 2 # 문서 임베딩

class StateTypeDict(TypedDict):
    documents : list
    similarity_documents : list
    vectorstore : Chroma
    user_query : str
    llm_answer : str
    execution_time_ms : float
    prompt_template : PromptTemplate
    state_result_msg : str
    state_result_type : StateResultType
    graph_query_type : GraphQueryType

class ErrorHandler:
    @staticmethod
    def check(state : StateTypeDict):
        return True
    
class Embedding_Helper:
    @staticmethod
    def make_vector_store_db(documents_source : list):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        all_splits = text_splitter.split_documents(documents_source)

        oembed = OllamaEmbeddings(base_url=ollama_service_url, model=ollama_model_name)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory=vectorstore_db_path)
        
        return f"기존에 존재하는 db가 없으므로 신규 생성합니다."
    
    @staticmethod
    def exists_vector_db():
        return os.path.exists(vectorstore_db_path)

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

    graph_query_type = state['graph_query_type']

    if graph_query_type == GraphQueryType.USER_REQUEST:
        state['documents'] = []
        state['state_result_type'] = StateResultType.SUCCESS
        state['state_result_msg'] = f"유저 쿼리이므로 문서 로딩을 하지 않습니다."
        state['execution_time_ms'] = 0.0

    elif graph_query_type == GraphQueryType.DOCS_EMBEDDING:

        performance_helper.start_timer()

        txt_loader = DirectoryLoader(path=documents_path, glob='**/*.txt')
        txt_files = txt_loader.load()

        total_data = []

        total_data += txt_files
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

        execution_ms, execution_sec = performance_helper.end_timer()
        state['execution_time_ms'] = execution_ms

        file_type_string = f"txt : {len(txt_files)}, pdf : {len(pdf_files)}, docx : {len(docx_files)}, doc : {len(doc_files)}, xlsx : {len(xlsx_files)}"

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

    performance_helper.start_timer()

    documents_source = state['documents']

    vectorstore_process_msg = ""

    if not Embedding_Helper.exists_vector_db():
        vectorstore_process_msg = Embedding_Helper.make_vector_store_db(documents_source)
    else:
        vectorstore = Chroma(persist_directory=vectorstore_db_path, embedding_function=OllamaEmbeddings(base_url=ollama_service_url, model=ollama_model_name))

        collected_docs = []
        #for index in range(len(vectorstore.get()["ids"])):
        #    doc_metadata = vectorstore.get()["metadatas"][index]
        #    collected_docs.append(doc_metadata["source"])

        new_docs = []
        #for data in documents_source:
        #    new_docs.append(data.metadata['source'])

        is_new_docs_subset = set(new_docs).issubset(set(collected_docs))

        if not is_new_docs_subset:
            vectorstore_process_msg = Embedding_Helper.make_vector_store_db(documents_source)
        else:
            vectorstore_process_msg = f"신규 문서가 없으므로, 기존 임베딩된 벡터스토어 DB를 사용합니다."

    execution_ms, execution_sec = performance_helper.end_timer()
    state['execution_time_ms'] = execution_ms

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

    performance_helper.start_timer()

    result_type = StateResultType.NONE

    vectorstore = state['vectorstore']
    question = state['user_query']

    # 유사 문서를 최대 몇개까지 찾을지? => k 값.
    # - 낮을 수록 정확도가 높은 문서만을 고른다. 
    # - 높을 수록 포괄적인 의미를 가진 문서도 포함한다. 
    result_docs = vectorstore.similarity_search(question, k=2)

    #for doc in result_docs:
    #    print(f"similarity searched document content : {doc.page_content}")

    execution_ms, execution_sec = performance_helper.end_timer()
    state['execution_time_ms'] = execution_ms

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

def state_make_prompt_template(state : StateTypeDict):
 
    performance_helper.start_timer()

    custom_template = '''
    You are an intelligent assistant.
    Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

    Do not try to make up an answer:
    - If the answer to the question cannot be determined from the context alone, just say "문의하신 내용에 대한 정확한 답변을 드리기 어렵습니다. \n 관련 내용을 찾으려면 https://google.com 을 이용해주세요."
    - If the context is empty, just say "문의하신 내용에 대한 답변이 불가능합니다. \n 관련 내용을 찾으려면 https://google.com 을 이용해주세요."

    CONTEXT:
    {context}

    QUESTION:
    {input}
    
    '''
    result_prompt = PromptTemplate(
        template=custom_template
    )

    execution_ms, execution_sec = performance_helper.end_timer()
    state['execution_time_ms'] = execution_ms

    result_type = StateResultType.NONE
    if result_prompt == None:
        result_type = StateResultType.FAIL_MAKE_PROMPT_TEMPLATE
        result_msg = f"프롬프트 생성에 실패했습니다."
    else:
        result_type = StateResultType.SUCCESS
        result_msg = f"프롬프트 생성에 성공했습니다."

    state['state_result_type'] = result_type
    state['state_result_msg'] = result_msg
    state['prompt_template'] = result_prompt

    return state

def state_query(state : StateTypeDict):

    performance_helper.start_timer()

    vectorstore = state['vectorstore']
    question = state['user_query']
    prompt_template = state['prompt_template']

    result_msg = ""
    result_type = StateResultType.NONE

    response = None
    try:
        ollama = Ollama(base_url=ollama_service_url, model=ollama_model_name)
        combine_docs_chain = create_stuff_documents_chain(ollama, prompt_template)
        retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
        
        response = retrieval_chain.invoke({"input": question}) # response['answer']
        result_type = StateResultType.SUCCESS
        result_msg = f"쿼리 처리 성공."

    except Exception as e:
        result_type = StateResultType.FAIL_QUERY_ERROR
        result_msg = f"쿼리 처리 중 에러 발생: {e}"

    execution_ms, execution_sec = performance_helper.end_timer()
    state['execution_time_ms'] = execution_ms

    if (response != None) and (response['answer'] != None):
        state['llm_answer'] = response['answer']
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
    graph.add_node(search_similarity_documents_node_name, state_search_similarity_documents)
    graph.add_node(make_prompt_template_node_name, state_make_prompt_template)
    graph.add_node(query_node_name, state_query)

    # 노드 연결
    # ( START -> load -> embedding -> search(documents) -> make prompot -> (apply prompt) query -> END 흐름으로 처리)
    graph.add_edge(START, load_node_name)
    graph.add_edge(load_node_name, embedding_node_name)
    graph.add_edge(embedding_node_name, search_similarity_documents_node_name)
    graph.add_edge(search_similarity_documents_node_name, make_prompt_template_node_name)
    graph.add_edge(make_prompt_template_node_name, query_node_name)
    graph.add_edge(query_node_name, END)

    # 컴파일. ( 그래프에 설정된 노드, 엣지, 컨디셔널등을 컴파일한다. )
    app = graph.compile()
    return app

def process_rag_system(in_user_query : str, in_graph_query_type : GraphQueryType):
    # build graph
    app_result = build_stategraph()

    # 유저 쿼리 / 그래프 쿼리 타입 설정.
    inputs = {'user_query' : in_user_query, 'graph_query_type' :  in_graph_query_type}

    for output in app_result.stream(inputs):
        # 출력된 결과에서 키와 값을 순회합니다.
        for key, value in output.items():

            SimpleLogger.Log(f"==================================================", LogType.LANGGRAPH)
            SimpleLogger.Log(f"Output from node name : {key}", LogType.LANGGRAPH)
            
            message = value['state_result_msg']
            type = value['state_result_type']
            execution_time_ms = value['execution_time_ms']

            SimpleLogger.Log(f"state_result_msg : {message}, type : {type}, execution_time : {execution_time_ms:.3f}(ms)", LogType.LANGGRAPH)

            # 쿼리 노드인 경우, 유저 질의와 llm의 대답을 출력한다.
            if key == query_node_name:
                result_llm_answer = value.get('llm_answer')
                result_user_query = value.get('user_query')

                SimpleLogger.Log(f"user_query : {result_user_query}", LogType.LANGGRAPH)
                SimpleLogger.Log(f"llm_answer : {result_llm_answer}", LogType.LANGGRAPH)

                return result_llm_answer
            
# 타임아웃을 적용한 RAG 시스템 처리 함수
async def process_rag_system_with_timeout(in_user_query: str, in_timeout: int = default_time_out_sec):
    try:
        # asyncio.wait_for를 사용하여 타임아웃 적용
        llm_answer = await asyncio.wait_for(asyncio.to_thread(process_rag_system, in_user_query, GraphQueryType.USER_REQUEST), timeout=in_timeout)
        return llm_answer
    except asyncio.TimeoutError:
        return f"llm 응답 생성에 실패했습니다. 다시 시도 해주세요!"

# API 엔드포인트에서 타임아웃 적용
@api_app_server.post("/query")
async def query_api_process(query: QueryRequest):
    SimpleLogger.Log(f"query_api : user_query : {query.user_query}", LogType.API_APP)
    try:
        # 비동기 타임아웃 적용
        llm_answer = await process_rag_system_with_timeout(query.user_query, default_time_out_sec)
        SimpleLogger.Log(f"llm(+rag) answer : {llm_answer}", LogType.API_APP)
        return {"llm_answer": llm_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"웹서버 응답 실패 : {str(e)}")

@api_app_server.on_event("shutdown")
async def shutdown_event():
    SimpleLogger.Log(f"Server is shutting down...", LogType.API_APP)
    # 여기서 진행 중인 작업을 완료하거나 대기 시간을 설정할 수 있습니다
    await asyncio.sleep(2)  # 2초 동안 대기 (작업 완료를 위해)
    SimpleLogger.Log(f"Shutdown complete.", LogType.API_APP)

@api_app_server.get("/", response_class=HTMLResponse)
async def root_api_process():
    SimpleLogger.Log(f"rag app server root! ", LogType.API_APP)
     # HTML 콘텐츠를 반환합니다.
    html_content = """
    <html>
        <head>
            <title>RAG App Server</title>
        </head>
        <body>
            <h1>Welcome to the RAG App Server!</h1>
            <p>This is a simple FastAPI web page.</p>
        </body>
    </html>
    """
    return html_content