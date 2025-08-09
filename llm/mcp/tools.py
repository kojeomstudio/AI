"""
Declarative tools for the MCP server.

Usage:
  - `mcp_server` is imported from mcp_server.py and used to register tools.
  - Tools can be sync or async. FastMCP will run them appropriately.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from mcp_server import mcp_server

@mcp_server.tool
async def greet(name: str) -> str:
    """
    Simple async tool. Demonstrates coroutine support.
    """
    await asyncio.sleep(0)  # yield control (example of async)
    return f"Hello, {name}!"

@mcp_server.tool(
    name="find_products",
    description="Search the product catalog with optional category filtering.",
    tags={"catalog", "search"},
    meta={"version": "1.2", "author": "product-team"},
)
async def search_products_implementation(query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Example async tool.
    """
    await asyncio.sleep(0)  # simulate async I/O
    # Fake dataset
    data = [
        {"id": 1, "name": "Red Apple", "category": "fruit"},
        {"id": 2, "name": "Blueberry Muffin", "category": "bakery"},
        {"id": 3, "name": "Apple Pie", "category": "bakery"},
    ]
    q = (query or "").lower()
    result = [item for item in data if q in item["name"].lower()]
    if category:
        result = [item for item in result if item["category"].lower() == category.lower()]
    return result
