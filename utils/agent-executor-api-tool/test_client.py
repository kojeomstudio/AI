"""
Simple test client for Agent Executor API
"""
import requests
import json
from typing import Dict, Any


class AgentExecutorClient:
    """Client for interacting with Agent Executor API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client with base URL"""
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_agents(self) -> Dict[str, Any]:
        """List supported agent types"""
        response = requests.get(f"{self.base_url}/agents")
        response.raise_for_status()
        return response.json()

    def execute_agent(
        self,
        agent_type: str,
        command_args: str,
        timeout: int = 300,
        working_directory: str = None
    ) -> Dict[str, Any]:
        """
        Execute a coding agent

        Args:
            agent_type: Type of agent (e.g., 'claude-code')
            command_args: Command line arguments
            timeout: Timeout in seconds
            working_directory: Working directory path

        Returns:
            Execution result dictionary
        """
        payload = {
            "agent_type": agent_type,
            "command_args": command_args,
            "timeout": timeout
        }

        if working_directory:
            payload["working_directory"] = working_directory

        response = requests.post(
            f"{self.base_url}/execute",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def main():
    """Test the API with sample requests"""
    client = AgentExecutorClient()

    print("=" * 60)
    print("Agent Executor API - Test Client")
    print("=" * 60)
    print()

    # Test 1: Health check
    print("1. Health Check")
    print("-" * 60)
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print("✓ Health check passed")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
    print()

    # Test 2: List supported agents
    print("2. List Supported Agents")
    print("-" * 60)
    try:
        agents = client.list_agents()
        print(f"Supported agents: {', '.join(agents['supported_agents'])}")
        print("✓ Agent list retrieved")
    except Exception as e:
        print(f"✗ Failed to get agent list: {e}")
    print()

    # Test 3: Execute agent (example - will fail if agent not installed)
    print("3. Execute Agent (Test)")
    print("-" * 60)
    print("Note: This test will fail if the agent is not installed locally")
    try:
        result = client.execute_agent(
            agent_type="claude-code",
            command_args="--help",
            timeout=30
        )
        print(f"Success: {result['success']}")
        print(f"Exit Code: {result['exit_code']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        if result['output']:
            print(f"Output (first 200 chars):")
            print(result['output'][:200])
        if result['error']:
            print(f"Error: {result['error'][:200]}")
        print("✓ Agent execution completed")
    except requests.exceptions.HTTPError as e:
        print(f"✗ Agent execution failed (HTTP error): {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"✗ Agent execution failed: {e}")
    print()

    print("=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
