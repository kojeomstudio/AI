import pytest
from ollama import Client

@pytest.fixture
def llm_client():
    """ LLM 클라이언트 초기화 """
    return Client(
        host='http://localhost:11434',
        headers={'x-some-header': 'some-value'}
    )

@pytest.mark.parametrize("user_query", [
    "Hello, how are you?",
    "What is the capital of France?",
    "Translate 'apple' into Spanish."
])
def test_llm_response(llm_client, user_query):
    """ LLM 응답이 정상적으로 반환되는지 검증 """
    try:
        response = llm_client.chat(model='qwen2.5-cder:latest', messages=[
            {'role': 'user', 'content': user_query}
        ])
        
        assert 'message' in response, "Response missing 'message' key"
        assert 'content' in response['message'], "Response missing 'content' key"
        assert len(response['message']['content'].strip()) > 0, "Response content is empty"

        print(f"✅ Test passed for query: {user_query}")
    
    except Exception as e:
        pytest.fail(f"❌ Test failed for query: {user_query}, Error: {e}")
