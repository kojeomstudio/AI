{
  "mcpServers": {
    "serena": {
      # sse의 경우 url
      "url": "http://127.0.0.1:9121/sse", 
      "timeout": 30000,
      "trust": true
    },
    "serena": {
      # http의 경우 httpUrl로 설정.
      "httpUrl": "http://127.0.0.1:9121/mcp",
      "timeout": 30000,
      "trust": true
    }
  },
  "security": {
    "auth": {
      "selectedType": "oauth-personal"
    }
  }
}


# win, macos
사용자 홈 디렉터리에서 .gemini 폴더 settings.json 파일의 내용을 위 내용처럼 구성하면 된다.