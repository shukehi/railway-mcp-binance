"""Minimal FastAPI wrapper around an MCP server."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Declare MCP server definitions
# ---------------------------------------------------------------------------
server = Server(name="hello-mcp")

BINANCE_FAPI = "https://fapi.binance.com"
VALID_INTERVALS = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}
KLINE_KEYS = [
    "openTime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closeTime",
    "quoteAssetVolume",
    "numberOfTrades",
    "takerBuyBase",
    "takerBuyQuote",
    "ignore",
]


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
        ),
        Tool(
            name="get_binance_klines",
            description="Fetch Binance USDT-M futures candlestick data via /fapi/v1/klines.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol, e.g. BTCUSDT.",
                    },
                    "interval": {
                        "type": "string",
                        "enum": sorted(list(VALID_INTERVALS)),
                        "description": "Binance interval identifier.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1500,
                        "description": "Number of klines to return (default 500).",
                    },
                    "startTime": {
                        "type": "integer",
                        "description": "Millisecond timestamp for the start of the range.",
                    },
                    "endTime": {
                        "type": "integer",
                        "description": "Millisecond timestamp for the end of the range.",
                    },
                },
                "required": ["symbol", "interval"],
                "additionalProperties": False,
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None = None):
    """Handle tool invocation requests."""
    payload: Dict[str, Any] = arguments or {}

    if name == "say_hello":
        return _handle_say_hello(payload)
    if name == "get_binance_klines":
        return await _handle_get_binance_klines(payload)

    raise ValueError(f"Unknown tool: {name}")


def _handle_say_hello(payload: Dict[str, Any]) -> List[TextContent]:
    raw_name = payload.get("name")
    person = str(raw_name).strip() if raw_name is not None else "world"
    if not person:
        person = "world"
    greeting = f"Hello, {person}!"
    return [TextContent(type="text", text=greeting)]


async def _handle_get_binance_klines(payload: Dict[str, Any]) -> List[TextContent]:
    symbol = payload.get("symbol")
    interval = payload.get("interval")
    limit = payload.get("limit", 500)
    start_time = payload.get("startTime")
    end_time = payload.get("endTime")

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("'symbol' 必须是非空字符串，例如 BTCUSDT")
    if not isinstance(interval, str) or interval not in VALID_INTERVALS:
        raise ValueError(f"'interval' 必须是 {sorted(list(VALID_INTERVALS))} 之一")

    try:
        limit_int = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("'limit' 必须是整数") from exc
    if not 1 <= limit_int <= 1500:
        raise ValueError("'limit' 取值范围为 1-1500")

    start_ms = _normalize_optional_int(start_time, "startTime")
    end_ms = _normalize_optional_int(end_time, "endTime")

    try:
        klines = await fetch_binance_klines(
            symbol=symbol,
            interval=interval,
            limit=limit_int,
            start_time=start_ms,
            end_time=end_ms,
        )
    except httpx.HTTPStatusError as exc:
        raise ValueError(
            f"Binance API 返回错误 {exc.response.status_code}: {exc.response.text.strip()}"
        ) from exc
    except httpx.RequestError as exc:
        raise ValueError(f"请求 Binance API 失败: {exc}") from exc

    summary = {
        "symbol": symbol.upper(),
        "interval": interval,
        "count": len(klines),
        "range": {
            "openTime": klines[0]["openTime"] if klines else None,
            "closeTime": klines[-1]["closeTime"] if klines else None,
        },
    }

    payload = {"summary": summary, "klines": klines}
    text = json.dumps(payload, ensure_ascii=False)
    return [TextContent(type="text", text=text)]


def _normalize_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' 必须是整数") from exc
    if integer < 0:
        raise ValueError(f"'{field_name}' 不能为负数")
    return integer


async def fetch_binance_klines(
    *,
    symbol: str,
    interval: str,
    limit: int,
    start_time: int | None,
    end_time: int | None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{BINANCE_FAPI}/fapi/v1/klines", params=params)
        response.raise_for_status()
        raw_data = response.json()

    if not isinstance(raw_data, list):
        raise ValueError("Binance API 返回了异常数据格式")

    klines: List[Dict[str, Any]] = []
    for row in raw_data:
        if not isinstance(row, list) or len(row) < len(KLINE_KEYS):
            continue
        item = dict(zip(KLINE_KEYS, row))
        item["openTime"] = int(item["openTime"])
        item["closeTime"] = int(item["closeTime"])
        item["numberOfTrades"] = int(item["numberOfTrades"])
        numeric_fields = (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quoteAssetVolume",
            "takerBuyBase",
            "takerBuyQuote",
        )
        for key in numeric_fields:
            item[key] = float(item[key])
        item.pop("ignore", None)
        klines.append(item)

    return klines


# ---------------------------------------------------------------------------
# FastAPI integration utilities
# ---------------------------------------------------------------------------


def _parse_csv_env(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_transport_security_settings() -> TransportSecuritySettings:
    """Create security settings for DNS rebinding protection."""

    allowed_hosts = _parse_csv_env(os.getenv("MCP_ALLOWED_HOSTS"))
    render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
    if render_host:
        allowed_hosts.append(render_host.strip())

    allowed_origins = _parse_csv_env(os.getenv("MCP_ALLOWED_ORIGINS"))
    if not allowed_origins:
        allowed_origins = ["https://chatgpt.com", "https://chat.openai.com"]

    # Ensure uniqueness while preserving deterministic order
    deduped_hosts = list(dict.fromkeys(host for host in allowed_hosts if host))
    deduped_origins = list(dict.fromkeys(origin for origin in allowed_origins if origin))

    enable_protection = bool(deduped_hosts or deduped_origins)

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=enable_protection,
        allowed_hosts=deduped_hosts,
        allowed_origins=deduped_origins,
    )


manager = StreamableHTTPSessionManager(
    app=server,
    json_response=True,
    stateless=False,
    security_settings=_build_transport_security_settings(),
)


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
