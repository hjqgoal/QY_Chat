"""Launch with: python -m chat_reply_agent.mcp_server (stdio, no listening port)."""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from chat_reply_agent.services.weather_contract import WEATHER_DESCRIPTION, WeatherQuery
from chat_reply_agent.services.weather_service import query_weather


mcp = FastMCP("chat-reply-weather", log_level="WARNING")


@mcp.tool(description=WEATHER_DESCRIPTION, annotations=ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True,
))
def get_weather(query: WeatherQuery) -> dict:
    return query_weather(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
