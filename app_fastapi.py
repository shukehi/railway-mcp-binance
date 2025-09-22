import os
import json
import time
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel, Field, field_validator

from mcp.server import FastMCP
from mcp.types import TextContent

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

# Create MCP server using FastMCP
mcp_server = FastMCP(name=APP_NAME)

@mcp_server.tool(
    "get_binance_klines",
    description="Fetch Binance USDT-M perpetual candlesticks via /fapi/v1/klines."
)
def get_binance_klines(
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

# Create FastAPI app
app = FastAPI(title="Binance MCP Server", description="MCP server for Binance perpetual contract klines")

# Add logging middleware for debugging
@app.middleware("http")
async def log_mcp_requests(request: Request, call_next):
    if request.url.path.startswith("/mcp") and request.method == "POST":
        try:
            body = await request.body()
            body_str = body.decode("utf-8")
            print(f"MCP POST {request.url.path} - Body: {body_str[:500]}...")

            # Parse and log key info
            try:
                json_body = json.loads(body_str)
                method = json_body.get("method", "unknown")
                params = json_body.get("params", {})
                print(f"MCP Method: {method}, Params keys: {list(params.keys()) if isinstance(params, dict) else type(params)}")
            except Exception as e:
                print(f"Failed to parse MCP JSON: {e}")

        except Exception as e:
            print(f"Failed to read MCP request body: {e}")

    response = await call_next(request)
    return response

# Health check endpoint
@app.get("/")
def root():
    return {"status": "ok", "service": "Binance MCP Server", "mcp_endpoint": "/mcp"}

@app.get("/healthz")
def health_check():
    return {"status": "healthy", "timestamp": int(time.time() * 1000)}

# Mount FastMCP's streamable HTTP app at /mcp
print(f"Mounting FastMCP server '{APP_NAME}' at /mcp")
app.mount("/mcp", mcp_server.streamable_http_app)
print("MCP server registered")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    print(f"Starting FastAPI server with MCP on {host}:{port}")
    uvicorn.run(app, host=host, port=port)