import asyncio
import os

from langchain_ollama import ChatOllama
from mcp_use import MCPAgent, MCPClient

async def main():

 # Create configuration dictionary
    config = {
      "mcpServers": {
        "serena": {
            "url": "http://localhost:9121/sse",
            "headers": {
                "Authorization": "Bearer ${AUTH_TOKEN}"
                }
            }
      }
    }

    # Create MCPClient from configuration dictionary
    client = MCPClient.from_dict(config)

# Initialize local Ollama model
    llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434/",  # Default Ollama URL
    temperature=0.7
    )

    agent = MCPAgent(llm=llm, client=client)

# Run the query
    result = await agent.run(
        "Activate the project, Activate onboarding, /serena_projects/AI, c#으로 작성된 게임 서버 코드를 모두 검색해서 알려주세요. 반드시 한글로 답변해주세요.",
        max_steps= 10,
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())