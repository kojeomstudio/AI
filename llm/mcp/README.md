# MCP Server/Client (Refactor)

## What changed
- Server reads `config.json`, registers `ping`, imports `tools` before run.
- Client supports **Streamable-HTTP & HTTP** with **auto** negotiation, tolerant **SSE/JSON** parsing, and session handling.

## Files
- `mcp_server.py` – FastMCP server (JSON-RPC over HTTP). Reads `config.json`.
- `tools.py` – Tool definitions (sync OK; async도 지원됨).
- `mcp_client.py` – JSON-RPC client (aiohttp). Auto-detect transport mode.
- `config.json` – server/client settings.

## Run
```bash
pip install fastmcp aiohttp

# 1) Start server
python mcp_server.py

# 2) In another terminal, run client demo
python mcp_client.py
```

## Config
```json
{
  "server": {
    "transport": "http",
    "host": "0.0.0.0",
    "port": 4200,
    "path": "/mcp",
    "log_level": "debug"
  },
  "client": {
    "server_url": "http://127.0.0.1:4200/mcp/",
    "request_timeout_seconds": 15,
    "transport_mode": "auto"
  }
}
```

## Notes
- Client always sets `Accept: application/json, text/event-stream` to satisfy negotiation.
- Streamable-HTTP requires `Mcp-Session-Id` (obtained from `initialize` response headers).
- For plain HTTP, session header is omitted; JSON single responses are handled.
