

# windows os
# C:\Users\[UserName]\.codex

[projects.'\\?\C:\Workspaces\AI']
trust_level = "trusted"

[projects.'\\?\C:\Workspaces\Unity3DWorkSpace\HELLO_MY_WORLD']
trust_level = "trusted"

[mcp_servers.serena]
command = "docker"
args = [
  "exec", "-i", "serena-serena-win64-1",
  "uv", "run", "--directory", "/serena_projects",
  "serena-mcp-server", "--transport", "stdio"
]

[mcp_servers.my_remote_server]
type = "http"
url = "https://my-mcp-server.example.com/mcp"
headers = { Authorization = "Bearer YOUR_API_TOKEN" }

[mcp_servers.serena]
type = "http"
url = "http://localhost:9121/mcp"