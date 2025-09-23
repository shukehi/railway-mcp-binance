"""Minimal FastAPI wrapper around an MCP server."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI
from difflib import SequenceMatcher

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.server.session import ServerSession, InitializationState
import mcp.types as mcp_types
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

SEARCH_TOOL_NAMES = {"search", "search_action"}
DEFAULT_SEARCH_LIMIT = 5


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose available MCP tools."""
    return [
        Tool(
            name="search",
            description="Search Binance USDT-M futures metadata by symbol or pair.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords, e.g. BTC or perpetual.",
                    },
                    "topK": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of matches to return (default 5).",
                    },
                    "timeRange": {
                        "type": "string",
                        "enum": ["24h", "7d", "30d", "all"],
                        "description": "Optional temporal hint; currently informational only.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
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

    if name in SEARCH_TOOL_NAMES:
        return await _handle_search(payload)
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


async def _handle_search(payload: Dict[str, Any]) -> List[TextContent]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' 必须是非空字符串")

    top_k = payload.get("topK", DEFAULT_SEARCH_LIMIT)
    try:
        limit = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("'topK' 必须是整数") from exc
    if not 1 <= limit <= 10:
        raise ValueError("'topK' 取值范围为 1-10")

    time_range = payload.get("timeRange")
    if time_range is not None and time_range not in {"24h", "7d", "30d", "all"}:
        raise ValueError("'timeRange' 取值需为 24h/7d/30d/all 之一")

    try:
        results = await _search_binance_symbols(query.strip(), limit)
    except httpx.HTTPStatusError as exc:
        raise ValueError(
            f"Binance exchangeInfo 返回错误 {exc.response.status_code}: {exc.response.text.strip()}"
        ) from exc
    except httpx.RequestError as exc:
        raise ValueError(f"访问 Binance exchangeInfo 失败: {exc}") from exc

    payload_out: Dict[str, Any] = {
        "results": results,
        "query": query.strip(),
    }
    if time_range:
        payload_out["timeRange"] = time_range

    text = json.dumps(payload_out, ensure_ascii=False)
    return [TextContent(type="text", text=text)]


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


async def _search_binance_symbols(query: str, limit: int) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo")
        response.raise_for_status()
        data = response.json()

    symbols = data.get("symbols")
    if not isinstance(symbols, list):
        return []

    query_upper = query.upper()
    scored: list[tuple[float, Dict[str, Any]]] = []

    for entry in symbols:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol", "")
        pair = entry.get("pair", "")
        contract = entry.get("contractType", "")

        field_values = [symbol, pair, contract, entry.get("deliveryDate", "")]
        matches = [val for val in field_values if isinstance(val, str) and val]
        if not matches:
            continue

        score = 0.0
        for value in matches:
            value_upper = value.upper()
            if query_upper in value_upper:
                score = max(score, 0.6)
            score = max(score, SequenceMatcher(None, query_upper, value_upper).ratio())

        if score <= 0.2:
            continue

        title = symbol or pair or "Unknown symbol"
        url_symbol = symbol or pair
        snippet_parts = [
            f"Pair: {pair}" if pair else None,
            f"Contract: {contract}" if contract else None,
            f"Status: {entry.get('status', 'UNKNOWN')}",
        ]
        snippet = " | ".join(part for part in snippet_parts if part)

        result = {
            "title": f"{title} (USDT-M futures)",
            "url": f"https://www.binance.com/en/futures/{url_symbol}" if url_symbol else None,
            "snippet": snippet or "Binance futures contract metadata",
            "metadata": {
                "symbol": symbol,
                "pair": pair,
                "contractType": contract,
                "marginAsset": entry.get("marginAsset"),
                "quoteAsset": entry.get("quoteAsset"),
            },
        }
        scored.append((score, result))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


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


def _parse_csv_env(value: str | None, *, normalize_host: bool = False) -> list[str]:
    if not value:
        return []
    items: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if normalize_host:
            host = _normalize_host(item)
            if host:
                items.append(host)
            continue
        items.append(item)
    return items


def _normalize_host(value: str) -> str | None:
    """Normalize host strings, stripping schemes or paths."""

    if not value:
        return None

    value = value.strip()
    if not value:
        return None

    if "://" in value:
        parsed = urlparse(value)
        host = parsed.netloc or parsed.path
    else:
        host = value

    # Remove any trailing path fragments that may remain
    host = host.split("/")[0].strip()
    return host or None


def _build_transport_security_settings() -> TransportSecuritySettings:
    """Create security settings for DNS rebinding protection."""

    allowed_hosts = _parse_csv_env(os.getenv("MCP_ALLOWED_HOSTS"), normalize_host=True)

    env_host_keys = (
        "RENDER_EXTERNAL_HOSTNAME",
        "RAILWAY_PUBLIC_DOMAIN",
        "RAILWAY_STATIC_URL",
        "RAILWAY_URL",
        "RAILWAY_HTTP_URL",
        "RAILWAY_PRIVATE_DOMAIN",
        "RAILWAY_APP_DOMAIN",
    )
    for key in env_host_keys:
        host = _normalize_host(os.getenv(key, ""))
        if host:
            allowed_hosts.append(host)

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
    stateless=True,
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
# ---------------------------------------------------------------------------
# Compatibility patches
# ---------------------------------------------------------------------------


_original_received_request = ServerSession._received_request


async def _received_request_with_auto_initialized(
    self: ServerSession, responder
) -> None:
    await _original_received_request(self, responder)
    if (
        getattr(responder, "request", None)
        and isinstance(responder.request.root, mcp_types.InitializeRequest)
        and self._initialization_state == InitializationState.Initializing
    ):
        self._initialization_state = InitializationState.Initialized


ServerSession._received_request = _received_request_with_auto_initialized
