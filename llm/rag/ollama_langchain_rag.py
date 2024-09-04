
# pip install
# -> langchain
# -> langchain_community
# pip install -U langchain-community ( if needed... )


from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

from langchain.llms import Ollama
from langchain.chains import RetrievalQA

ollama_model_name="EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_S.gguf:latest"

loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
data = loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) 
all_splits = text_splitter.split_documents(data)

oembed = OllamaEmbeddings(base_url="http://localhost:11434", model=ollama_model_name)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

# 백터 스토어에 쿼리를 한다. 그리고 그에 대한 결과.
question="What's the name of main character?"
docs = vectorstore.similarity_search(question)
print(f"docs len : {len(docs)}, docs -> {docs}")

ollama = Ollama(base_url='http://localhost:11434', model="gemma")
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
qachain.invoke({"query": question})

#llm = Ollama(
#    model=ollama_model_name, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
#)
#llm("The first man on the summit of Mount Everest, the highest peak on Earth, was ...")

