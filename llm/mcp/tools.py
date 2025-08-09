import mcp_server

# Sample tool definitions for the MCP server
@mcp_server.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Sample tool with custom name, description, tags, and metadata
@mcp_server.tool(
    name="find_products",  # Custom tool name for the LLM
    description="Search the product catalog with optional category filtering.",
    tags={"catalog", "search"},
    meta={"version": "1.2", "author": "product-team"},
)
def search_products_implementation(query: str, category: str | None = None) -> list[dict]:
    # Minimal fake search over a toy in-memory dataset
    data = [
        {"id": 1, "name": "Red Apple", "category": "fruit"},
        {"id": 2, "name": "Blueberry Muffin", "category": "bakery"},
        {"id": 3, "name": "Apple Pie", "category": "bakery"},
    ]
    q = (query or "").lower()
    out = [item for item in data if q in item["name"].lower()]
    if category:
        out = [item for item in out if item["category"].lower() == category.lower()]
    return out
