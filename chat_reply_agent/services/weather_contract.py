"""Shared MCP/tool contract; importing this does not launch a server or read secrets."""

import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WeatherQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    city: str = Field(min_length=2, max_length=80, description="聊天明确提到的城市；不能根据语言、IP或历史居住地猜当前地点")
    country_code: str = Field(default="", pattern=r"^([A-Z]{2})?$", description="明确知道时填写国家代码，例如 CN；否则留空")
    region: str = Field(default="", max_length=80, description="明确知道的省/州名称，用于消除同名城市歧义")
    day: str = Field(default="today", max_length=10, description="today、tomorrow 或 YYYY-MM-DD；以查询地点当地日期为准，只支持未来7天")
    period: Literal["day", "morning", "afternoon", "evening"] = "day"

    @field_validator("city", "region")
    @classmethod
    def validate_place(cls, value):
        if any(not (character.isalnum() or character in " -.,'’()") for character in value):
            raise ValueError("Use a place name, not a URL, address or message")
        return value

    @field_validator("day")
    @classmethod
    def validate_day(cls, value):
        if value not in {"today", "tomorrow"}:
            parsed = date.fromisoformat(value)
            if parsed.isoformat() != value:
                raise ValueError("Use YYYY-MM-DD")
        return value


WEATHER_DESCRIPTION = (
    "查询 Open-Meteo 天气及空气质量，供穿衣、出行、温度湿度、雾、大风等回复使用。"
    "只在本轮确实需要天气依据且城市明确时调用。只传城市/地区/日期，不传聊天原文。"
    "返回地点、当地时间、预报时段、温度/体感、湿度、降水、风速阵风及 PM2.5/PM10/美国AQI。"
    "地点有歧义时返回候选，必须依据用户已给信息消歧，否则询问用户；不能自行挑选。"
    "数据是模型估计/预报，不是实测或官方预警。缺失数据不能当成零。"
)


def weather_tool_definition():
    return {"type": "function", "function": {
        "name": "get_weather", "description": WEATHER_DESCRIPTION,
        "parameters": {"type": "object", "properties": {"query": WeatherQuery.model_json_schema()},
                       "required": ["query"], "additionalProperties": False},
    }}


def weather_evidence_context(results: list[dict]) -> str:
    if not results:
        return ""
    return ("\n外部天气工具结果（仅本轮使用，不是聊天原话）：\n"
            + json.dumps(results, ensure_ascii=False)
            + "\n只引用成功获取且对应地点/日期/时段的数据。partial 要说明缺项，失败或歧义时不要编造天气。"
              "AQI 是美国标准，不是中国AQI；湿度高、雾与霾不能混为一谈。穿衣建议要按体感、风雨及个人情况表达，"
              "不要声称官方预警、实时观测或把未来预报说成已发生。回复自然接话，不堆砌数字。")
