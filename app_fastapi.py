# app_fastapi.py - 最小可运行版本
from fastapi import FastAPI
from mcp.server import Server
from mcp.server.http import create_asgi_app
from mcp.types import TextContent

# 1. 创建 MCP Server
srv = Server(name="hello-mcp")

# 2. 注册一个最简单的工具
@srv.tool("say_hello", description="Return a hello message")
def say_hello(name: str = "world"):
    return [TextContent(type="text", text=f"Hello, {name}!")]

# 3. 创建 FastAPI 应用，并挂载 MCP 路径
api = FastAPI(title="Hello MCP Server")
api.mount("/mcp", create_asgi_app(srv))
