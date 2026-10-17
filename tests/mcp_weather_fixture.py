"""Deterministic stdio MCP server fixture. Never contacts Open-Meteo."""

import os
import time

from chat_reply_agent import mcp_server


def fake_weather(query):
    if query.city == "Timeout":
        time.sleep(20)
    return {"status": "ok", "location": {"name": query.city, "timezone": "Asia/Shanghai"},
            "provider": "offline-test", "day": query.day,
            "has_model_secret": bool(os.environ.get("OPENAI_API_KEY"))}


mcp_server.query_weather = fake_weather
if __name__ == "__main__":
    mcp_server.mcp.run(transport="stdio")
