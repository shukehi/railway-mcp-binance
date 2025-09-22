import os
import json
import time
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field, field_validator

from mcp.server import Server
from mcp.types import TextContent
from mcp.server.streamable_http import StreamableHTTPServerTransport
import asyncio

# === Config ===
APP_NAME = "binance-perp-kline"
BINANCE_FAPI = "https://fapi.binance.com"
VALID_INTERVALS = {
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","3d","1w","1M"
}

class KlineInput(BaseModel):
    symbol: str = Field(..., description="USDT-M perpetual symbol, e.g. 'BTCUSDT'")
    interval: str = Field(..., description="Binance interval")
    limit: int = Field(500, ge=1, le=1500, description="Number of candles (max 1500)")
    startTime: Optional[int] = Field(None, description="ms since epoch (UTC)")
    endTime: Optional[int] = Field(None, description="ms since epoch (UTC)")

    @field_validator("interval")
    @classmethod
    def _interval_ok(cls, v: str) -> str:
        if v not in VALID_INTERVALS:
            raise ValueError(f"interval must be one of {sorted(list(VALID_INTERVALS))}")
        return v

def fetch_klines(
    symbol: str, interval: str, limit: int = 500,
    start_ms: Optional[int] = None, end_ms: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Call Binance USDT-M perpetual klines /fapi/v1/klines (public)."""
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{BINANCE_FAPI}/fapi/v1/klines", params=params)
        resp.raise_for_status()
        raw = resp.json()

    keys = [
        "openTime","open","high","low","close","volume",
        "closeTime","quoteAssetVolume","numberOfTrades",
        "takerBuyBase","takerBuyQuote","ignore"
    ]
    out: List[Dict[str, Any]] = []
    for row in raw:
        item = dict(zip(keys, row))
        for k in ["open","high","low","close","volume","quoteAssetVolume","takerBuyBase","takerBuyQuote"]:
            if k in item:
                item[k] = float(item[k])
        out.append(item)
    return out

# Create MCP server using standard Server class
srv = Server(name=APP_NAME)

# Define tool function
def tool_get_binance_klines(
    symbol: str,
    interval: str,
    limit: int = 500,
    startTime: Optional[int] = None,
    endTime: Optional[int] = None
):
    """Return JSON text with summary + klines array."""
    # Validate user input
    inp = KlineInput(symbol=symbol, interval=interval, limit=limit, startTime=startTime, endTime=endTime)

    data = fetch_klines(
        symbol=inp.symbol,
        interval=inp.interval,
        limit=inp.limit,
        start_ms=inp.startTime,
        end_ms=inp.endTime
    )

    summary = {
        "symbol": inp.symbol.upper(),
        "interval": inp.interval,
        "count": len(data),
        "range": {
            "openTime_min": data[0]["openTime"] if data else None,
            "closeTime_max": data[-1]["closeTime"] if data else None
        },
        "generated_at": int(time.time() * 1000)
    }
    return [TextContent(type="text", text=json.dumps({"summary": summary, "klines": data}, ensure_ascii=False))]

# Register the tool with server
@srv.call_tool()
async def handle_call_tool(name, arguments):
    """Handle tool calls"""
    if name == "get_binance_klines":
        return tool_get_binance_klines(**arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

# Register tools list
@srv.list_tools()
async def handle_list_tools():
    """Return available tools"""
    from mcp.types import Tool
    return [
        Tool(
            name="get_binance_klines",
            description="Fetch Binance USDT-M perpetual candlesticks via /fapi/v1/klines.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "USDT-M perpetual symbol, e.g. 'BTCUSDT'"},
                    "interval": {"type": "string", "description": "Binance interval"},
                    "limit": {"type": "integer", "default": 500, "minimum": 1, "maximum": 1500, "description": "Number of candles (max 1500)"},
                    "startTime": {"type": "integer", "description": "ms since epoch (UTC)"},
                    "endTime": {"type": "integer", "description": "ms since epoch (UTC)"}
                },
                "required": ["symbol", "interval"]
            }
        )
    ]

# Create FastAPI app
api = FastAPI(
    title="Binance MCP Server",
    description="MCP server for Binance perpetual contract klines with proper session management"
)

# Health check endpoints
@api.get("/")
def root():
    return {"status": "ok", "service": "Binance MCP Server", "mcp_endpoint": "/mcp"}

@api.get("/healthz")
def health_check():
    return {"status": "healthy", "timestamp": int(time.time() * 1000)}

# Debug middleware for MCP requests
@api.middleware("http")
async def log_mcp_requests(request: Request, call_next):
    if request.url.path.startswith("/mcp") and request.method == "POST":
        try:
            body = await request.body()
            body_str = body.decode("utf-8")
            print(f"MCP POST {request.url.path}")
            print(f"Headers: {dict(request.headers)}")
            print(f"Body: {body_str[:500]}...")

            # Parse and log key info
            try:
                json_body = json.loads(body_str)
                method = json_body.get("method", "unknown")
                request_id = json_body.get("id", "no-id")
                print(f"MCP Method: {method}, ID: {request_id}")
            except Exception as e:
                print(f"Failed to parse MCP JSON: {e}")

        except Exception as e:
            print(f"Failed to read MCP request body: {e}")

    response = await call_next(request)

    # Log response headers for debugging
    if request.url.path.startswith("/mcp"):
        print(f"Response headers: {dict(response.headers)}")

    return response

# Create streamable HTTP transport for MCP server
transport = StreamableHTTPServerTransport(srv)

@api.post("/mcp/{path:path}")
async def mcp_handler(request: Request, path: str):
    """Handle MCP requests with proper session management"""
    # Get request body
    body = await request.body()

    # Create ASGI scope, receive, and send
    scope = {
        "type": "http",
        "method": request.method,
        "path": f"/mcp/{path}",
        "query_string": str(request.url.query).encode(),
        "headers": [(k.encode(), v.encode()) for k, v in request.headers.items()],
    }

    # Handle the request through streamable HTTP transport
    response_data = []
    response_headers = {}

    async def receive():
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    async def send(message):
        if message["type"] == "http.response.start":
            response_headers.update({
                k.decode(): v.decode() for k, v in message.get("headers", [])
            })
        elif message["type"] == "http.response.body":
            response_data.append(message.get("body", b""))

    # Process through transport
    await transport(scope, receive, send)

    # Build response
    response_body = b"".join(response_data)

    # Create FastAPI response with all headers
    response = Response(
        content=response_body,
        headers=response_headers,
        media_type=response_headers.get("content-type", "text/event-stream")
    )

    return response

print(f"MCP server '{APP_NAME}' configured with StreamableHTTPServerTransport")
print("Proper session management enabled")