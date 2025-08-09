"""
MCP Server (FastMCP-based)
- JSON-RPC 2.0 over HTTP
- Settings loaded from config.json
- Exposes `mcp_server` object for tool registration (used by tools.py)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from fastmcp import FastMCP

# Load config.json (server section)
CONFIG_PATH = Path(__file__).with_name("config.json")
if CONFIG_PATH.exists():
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = raw.get("server", {})
else:
    cfg = {}

TRANSPORT: str = cfg.get("transport", "http")
HOST: str = cfg.get("host", "0.0.0.0")
PORT: int = int(cfg.get("port", 4200))
PATH: str = cfg.get("path", "/mcp")
LOG_LEVEL: str = cfg.get("log_level", "info")

# Create MCP server instance (exported for tools.py to import)
mcp_server = FastMCP("MCP Server")

# ---- Optional: basic health/ping tool to check JSON-RPC wiring ----
@mcp_server.tool(
    name="ping",
    description="Liveness probe. Returns 'pong' with optional echo payload."
)
async def ping(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {"ok": True, "result": "pong", "echo": payload or {}}

# NOTE: Do not import tools here at module import time to avoid circular issues
# Tools should import `mcp_server` and register themselves. Then, when running
# as `__main__`, we import tools to ensure registration.

def _import_tools():
    try:
        import tools  # noqa: F401  (registration side-effects)
    except Exception as e:
        # Keep running even if tools are absent; log a warning at startup.
        print(f"[warn] tools import failed: {e}")

if __name__ == "__main__":
    # Ensure tools are registered before running
    _import_tools()

    # Start FastMCP with transport from config
    # For HTTP transport, JSON-RPC 2.0 requests go to http://HOST:PORT{PATH}
    mcp_server.run(
        transport=TRANSPORT,
        host=HOST,
        port=PORT,
        path=PATH,
        log_level=LOG_LEVEL,
    )
