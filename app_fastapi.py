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
manager = StreamableHTTPSessionManager(app=server, json_response=False, stateless=True)


class MCPASGIApp:
    """Minimal ASGI adapter that delegates to the MCP session manager."""

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self._manager = session_manager

    async def __call__(self, scope, receive, send):
        scope_type = scope.get("type")
        if scope_type == "http":
            scope = dict(scope)
            scope["headers"] = self._ensure_streamable_accept(scope.get("headers", []), scope.get("path", ""))
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

    @staticmethod
    def _ensure_streamable_accept(headers, path: str):
        """Ensure StreamableHTTP endpoints advertise JSON + SSE support."""
        if not path.startswith("/mcp"):
            return headers

        header_list = list(headers)
        accept_idx = -1
        current_values: list[str] = []

        for idx, (name, value) in enumerate(header_list):
            if name.lower() == b"accept":
                accept_idx = idx
                current = value.decode("latin-1")
                current_values = [item.strip() for item in current.split(",") if item.strip()]
                break

        required = ["application/json", "text/event-stream"]
        for item in required:
            if item not in current_values:
                current_values.append(item)

        new_accept = ", ".join(current_values) if current_values else ", ".join(required)
        encoded = new_accept.encode("latin-1")

        if accept_idx >= 0:
            header_list[accept_idx] = (b"accept", encoded)
        else:
            header_list.append((b"accept", encoded))

        return header_list


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
