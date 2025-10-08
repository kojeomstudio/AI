

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
