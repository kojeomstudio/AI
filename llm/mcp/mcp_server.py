"""
MCP Server (FastMCP-based)
- JSON-RPC 2.0 over HTTP (Streamable-HTTP in FastMCP 2.x)
- Settings loaded from config.json
- Tools registered via decorators in tools.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from fastmcp import FastMCP

# Load config
CONFIG_PATH = Path(__file__).with_name("config.json")
cfg = {"server": {}}
if CONFIG_PATH.exists():
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] Failed to parse config.json: {e}")

scfg = cfg.get("server", {})
TRANSPORT = scfg.get("transport", "http")
HOST = scfg.get("host", "0.0.0.0")
PORT = int(scfg.get("port", 4200))
PATH = scfg.get("path", "/mcp")  # base path (client should POST to /mcp/)
LOG_LEVEL = scfg.get("log_level", "info")

# Create server object for tools.py to import and register tools
mcp_server = FastMCP("MCP Server")

# Simple liveness probe (also helps verify JSON-RPC+session wiring)
@mcp_server.tool(name="ping", description="Liveness probe. Returns pong with optional echo payload.")
async def ping(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {"ok": True, "result": "pong", "echo": payload or {}}

def _import_tools():
    # Ensure tool registration side effects happen before server run
    try:
        import tools  # noqa: F401  (registration via decorators)
    except Exception as e:
        print(f"[warn] tools import failed (no user tools registered): {e}")

if __name__ == "__main__":
    _import_tools()
    mcp_server.run(
        transport=TRANSPORT,
        host=HOST,
        port=PORT,
        path=PATH,
        log_level=LOG_LEVEL,
    )
