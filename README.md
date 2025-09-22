# Binance Perp Klines — MCP Server (Railway-ready)

This is a minimal **MCP server** exposing a single tool:
- `get_binance_klines(symbol, interval, limit=500, startTime?, endTime?)`

It uses **HTTP/SSE** transport so **ChatGPT web** (and other MCP clients) can connect.

## 1) Local test
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PORT=8000
python server.py
# MCP endpoint: http://127.0.0.1:8000/mcp  (use a tunnel or deploy to test with ChatGPT web)
