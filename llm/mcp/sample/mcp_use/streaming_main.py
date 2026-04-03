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
        base_url="http://localhost:11434/",
        temperature=0.7
    )

    agent = MCPAgent(llm=llm, client=client)

    async for item in agent.stream("c#으로 작성된 게임 서버 코드를 모두 검색해서 알려주세요. 반드시 한글로 답변해주세요."):
        if isinstance(item, str):
            # Final result
            print(f"\nFinal Result:\n{item}")
        else:
            # Intermediate step (action, observation)
            action, observation = item
            print(f"\nTool: {action.tool}")
            print(f"Input: {action.tool_input}")
            print(f"Result: {observation[:100]}{'...' if len(observation) > 100 else ''}")

    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
