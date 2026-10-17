"""Exercise the real MCP weather tool, without invoking any language model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat_reply_agent.services.mcp_service import call_weather_tool


def main():
    parser = argparse.ArgumentParser(description="Query Open-Meteo through the bundled MCP server")
    parser.add_argument("city")
    parser.add_argument("--country", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--day", default="today")
    parser.add_argument("--period", choices=["day", "morning", "afternoon", "evening"], default="day")
    args = parser.parse_args()
    result = call_weather_tool({"query": {"city": args.city, "country_code": args.country,
        "region": args.region, "day": args.day, "period": args.period}})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
