from langchain_ollama import ChatOllama
from mcp_use import MCPAgent, MCPClient

# Initialize local Ollama model
llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0.7
)