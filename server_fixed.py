from __future__ import annotations
import os, time, json
from typing import Optional, List, Dict, Any

import httpx
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

if __name__ == "__main__":
    # Railway 注入 PORT 环境变量
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    print(f"Starting MCP server '{APP_NAME}' on {host}:{port}")
    print("Available tools: get_binance_klines")

    # Create FastMCP app with host and port
    app = FastMCP(name=APP_NAME, host=host, port=port)

    @app.tool(
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

    # 使用 SSE 传输模式，兼容ChatGPT和Claude客户端
    app.run(transport="sse")