"""Minimal FastAPI wrapper around an MCP server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Declare MCP server definitions
# ---------------------------------------------------------------------------
server = Server(name="hello-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose available MCP tools."""
    return [
        Tool(
            name="say_hello",
            description="Return a friendly greeting",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Optional name to include in the greeting.",
                    }
                },
                "required": [],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None = None):
    """Handle tool invocation requests."""
    if name != "say_hello":
        raise ValueError(f"Unknown tool: {name}")

    payload = arguments or {}
    raw_name = payload.get("name")
    person = str(raw_name).strip() if raw_name is not None else "world"
    if not person:
        person = "world"
    greeting = f"Hello, {person}!"
    return [TextContent(type="text", text=greeting)]


# ---------------------------------------------------------------------------
# FastAPI integration utilities
# ---------------------------------------------------------------------------
manager = StreamableHTTPSessionManager(app=server, json_response=False, stateless=False)


class MCPASGIApp:
    """Minimal ASGI adapter that delegates to the MCP session manager."""

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self._manager = session_manager

    async def __call__(self, scope, receive, send):
        scope_type = scope.get("type")
        if scope_type == "http":
            await self._manager.handle_request(scope, receive, send)
            return

        if scope_type == "lifespan":
            while True:
                message = await receive()
                message_type = message["type"]
                if message_type == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message_type == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            return

        raise RuntimeError(f"Unsupported ASGI scope type: {scope_type}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Ensure the MCP session manager is running for the FastAPI app lifespan."""
    async with manager.run():
        yield


api = FastAPI(title="Hello MCP Server", lifespan=lifespan)
api.mount("/mcp", MCPASGIApp(manager))


@api.get("/healthz")
async def healthz() -> dict[str, str]:
    """Simple health endpoint."""
    return {"status": "ok"}
