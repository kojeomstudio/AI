# MCP Server/Client (Refactor)


## Client transport modes
- `client.transport_mode`: `"auto"` (default), `"streamable-http"`, `"http"`
- Auto mode tries Streamable-HTTP first, then falls back to plain HTTP on 406 or missing-session issues.
- Streamable-HTTP: requires `Accept: application/json, text/event-stream` and `Mcp-Session-Id` after `initialize`.
