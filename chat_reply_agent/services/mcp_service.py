import asyncio
from collections import OrderedDict
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import sys
from threading import Lock
from time import monotonic
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from chat_reply_agent.config import get_settings
from chat_reply_agent.services.telemetry_service import timed_stage
from chat_reply_agent.services.weather_contract import WeatherQuery


_cache = OrderedDict()
_cache_lock = Lock()


def _same_local_day(data):
    try:
        zone = ZoneInfo(data["location"]["timezone"])
        fetched = datetime.fromisoformat(data["fetched_at"]).astimezone(zone).date()
        return fetched == datetime.now(zone).date()
    except (KeyError, TypeError, ValueError):
        return False


async def _call_mcp(arguments: dict) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = Path(__file__).resolve().parents[2]
    parameters = StdioServerParameters(command=sys.executable,
        args=["-m", "chat_reply_agent.mcp_server"], cwd=str(root),
        env={"PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"})
    timeout = get_settings().weather_tool_timeout_seconds
    # The fixed, bundled server receives location arguments, never prompts or API keys.
    async with asyncio.timeout(timeout):
        with open(os.devnull, "w") as error_log:
            async with stdio_client(parameters, errlog=error_log) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=timeout)) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    if "get_weather" not in {tool.name for tool in tools.tools}:
                        raise ValueError("Missing weather tool")
                    result = await session.call_tool("get_weather", arguments)
                    if result.isError:
                        raise ValueError("MCP tool failed")
                    data = result.structuredContent
                    if not isinstance(data, dict):
                        blocks = [block.text for block in result.content if block.type == "text"]
                        data = json.loads(blocks[0]) if blocks else None
                    if not isinstance(data, dict) or "status" not in data:
                        raise ValueError("Invalid MCP response")
                    return data


@timed_stage("tool.weather")
def call_weather_tool(arguments: dict) -> dict:
    if not get_settings().weather_tool_enabled:
        return {"status": "disabled", "message": "天气工具已关闭，不得声称查询过天气。"}
    try:
        if not isinstance(arguments, dict) or set(arguments) != {"query"}:
            raise ValueError("Expected query")
        query = WeatherQuery.model_validate(arguments["query"], strict=True)
    except (ValueError, ValidationError):
        return {"status": "invalid_arguments", "message": "请提供明确城市和有效日期；仅接受地点与日期参数。"}
    key = (query.model_dump_json(), datetime.now(UTC).date().isoformat())
    ttl = get_settings().weather_cache_seconds
    with _cache_lock:
        cached = _cache.get(key)
        if cached and monotonic() - cached[0] < ttl and _same_local_day(cached[1]):
            return {**deepcopy(cached[1]), "cached": True}
    try:
        data = asyncio.run(_call_mcp({"query": query.model_dump()}))
    except Exception:
        return {"status": "unavailable", "message": "MCP 天气查询未完成，请检查依赖或网络；不可编造天气。"}
    if data["status"] in {"ok", "partial"}:
        with _cache_lock:
            _cache[key] = (monotonic(), deepcopy(data))
            _cache.move_to_end(key)
            while len(_cache) > 128:
                _cache.popitem(last=False)
    return {**data, "cached": False}
