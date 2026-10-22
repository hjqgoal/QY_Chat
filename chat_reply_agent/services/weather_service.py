"""Read-only Open-Meteo adapter. Only fixed public endpoints receive location data."""

import math
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from chat_reply_agent.services.weather_contract import WeatherQuery


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FIELDS = ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
                  "precipitation", "weather_code", "wind_speed_10m", "wind_gusts_10m"]
HOURLY_FIELDS = WEATHER_FIELDS + ["precipitation_probability"]
DAILY_FIELDS = ["weather_code", "temperature_2m_max", "temperature_2m_min",
                "apparent_temperature_max", "apparent_temperature_min", "precipitation_probability_max",
                "wind_speed_10m_max", "wind_gusts_10m_max"]
AIR_FIELDS = ["pm2_5", "pm10", "us_aqi"]
SOURCES = [
    {"name": "Open-Meteo Weather", "url": "https://open-meteo.com/en/docs"},
    {"name": "Open-Meteo / CAMS Air Quality", "url": "https://open-meteo.com/en/docs/air-quality-api"},
    {"name": "CAMS", "url": "https://ads.atmosphere.copernicus.eu/"},
    {"name": "GeoNames", "url": "https://www.geonames.org/"},
]


def _get(client, url, params):
    response = client.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError("Invalid weather response")
    return data


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _location(item):
    return {key: item.get(key) for key in ("name", "admin1", "country", "country_code", "latitude", "longitude", "timezone")}


def _normal_name(value):
    text = str(value or "").strip().casefold()
    for suffix in ("自治区", "省", "市"):
        if text.endswith(suffix) and len(text) > len(suffix) + 1:
            return text[:-len(suffix)]
    return text


def _hour_rows(data, fields, target, period):
    if not isinstance(data, dict) or not isinstance(data.get("time", []), list):
        raise ValueError("Invalid hourly data")
    start, end = {"day": (0, 24), "morning": (6, 12), "afternoon": (12, 18), "evening": (18, 24)}[period]
    rows = []
    for index, stamp in enumerate(data.get("time", [])):
        moment = datetime.fromisoformat(stamp)
        if moment.date() == target and start <= moment.hour < end:
            row = {"time": stamp}
            for key in fields:
                values = data.get(key, [])
                row[key] = _number(values[index]) if isinstance(values, list) and index < len(values) else None
            rows.append(row)
    return rows


def _range(rows, field):
    numbers = [row[field] for row in rows if row[field] is not None]
    return {"min": min(numbers), "max": max(numbers), "available_hours": len(numbers)} if numbers else None


def _query(query, client, now):
    # GeoNames finds "北京", but returns no results for the official "北京市" form.
    search_name = query.city[:-1] if query.city.endswith("市") and len(query.city) > 2 else query.city
    region = query.region
    if not region and query.country_code == "CN" and query.city in {"北京市", "上海市", "天津市", "重庆市"}:
        region = query.city
    params = {"name": search_name, "count": 10, "language": "zh", "format": "json"}
    if query.country_code:
        params["countryCode"] = query.country_code
    candidates = _get(client, GEOCODING_URL, params).get("results", [])
    if not isinstance(candidates, list):
        raise ValueError("Invalid locations")
    candidates = [item for item in candidates if isinstance(item, dict)
                  and _number(item.get("latitude")) is not None and _number(item.get("longitude")) is not None]
    if query.country_code:
        candidates = [item for item in candidates if item.get("country_code") == query.country_code]
    if region:
        candidates = [item for item in candidates if _normal_name(item.get("admin1")) == _normal_name(region)]
    exact = [item for item in candidates if _normal_name(item.get("name")) == _normal_name(query.city)]
    if exact:
        candidates = exact
    if not candidates:
        return {"status": "not_found", "message": "未找到地点，请补充城市、省/州或国家，不能猜测位置。"}
    if len(candidates) != 1:
        return {"status": "needs_clarification", "message": "有多个同名地点，请依据已知地区消歧或向用户确认。",
                "locations": [_location(item) for item in candidates[:5]]}
    location = _location(candidates[0])
    zone = ZoneInfo(location["timezone"])
    local_now = now.astimezone(zone)
    target = (local_now.date() + timedelta(days=query.day == "tomorrow")
              if query.day in {"today", "tomorrow"} else date.fromisoformat(query.day))
    if not 0 <= (target - local_now.date()).days <= 6:
        return {"status": "unsupported_date", "message": "仅支持当地今天起7天内的预报，不能用今天数据代替历史或更远日期。",
                "location": location, "local_time": local_now.isoformat()}
    common = {"latitude": location["latitude"], "longitude": location["longitude"],
              "timezone": location["timezone"], "forecast_days": 7}
    data = _get(client, FORECAST_URL, {**common, "current": ",".join(WEATHER_FIELDS),
                "hourly": ",".join(HOURLY_FIELDS), "daily": ",".join(DAILY_FIELDS),
                "wind_speed_unit": "ms", "temperature_unit": "celsius", "precipitation_unit": "mm"})
    daily = data.get("daily", {})
    if not isinstance(daily, dict):
        raise ValueError("Invalid daily data")
    times = daily.get("time", [])
    daily_index = times.index(target.isoformat()) if target.isoformat() in times else None
    selected_daily = {key: _number(daily.get(key, [])[daily_index])
                      if daily_index is not None and len(daily.get(key, [])) > daily_index else None
                      for key in DAILY_FIELDS}
    rows = _hour_rows(data.get("hourly", {}), HOURLY_FIELDS, target, query.period)
    current = data.get("current", {})
    if not isinstance(current, dict):
        current = {}
    # Current estimates are separate from the requested day's forecast, never substitute them.
    current_weather = ({"time": current.get("time"), **{key: _number(current.get(key)) for key in WEATHER_FIELDS}}
                       if target == local_now.date() else None)
    warnings = ["天气和空气质量均为模型估计/预报，不代表实测或官方预警。", "美国AQI不是中国AQI；雾和霾不能仅凭湿度判断。"]
    air = None
    try:
        air_data = _get(client, AIR_URL, {**common, "hourly": ",".join(AIR_FIELDS)})
        air_rows = _hour_rows(air_data.get("hourly", {}), AIR_FIELDS, target, query.period)
        air = {"standard": "US AQI", "period_ranges": {key: _range(air_rows, key) for key in AIR_FIELDS}}
        if any(value is None for value in air["period_ranges"].values()):
            warnings.append("该时段部分空气质量数据不可用，不得推断为空气良好。")
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        warnings.append("空气质量查询失败，仅可引用已获取的天气数据。")
    ranges = {key: _range(rows, key) for key in HOURLY_FIELDS if key != "weather_code"}
    expected_hours = 24 if query.period == "day" else 6
    incomplete = (daily_index is None or any(value is None for value in selected_daily.values())
                  or any(value is None or value["available_hours"] < expected_hours for value in ranges.values()) or air is None
                  or any(value is None or value["available_hours"] < expected_hours for value in air["period_ranges"].values()))
    if incomplete:
        warnings.append("部分字段或小时缺失；范围仅反映已获取的数据，不保证覆盖整个时段。")
    if not any(value is not None for value in ranges.values()) and not any(value is not None for value in selected_daily.values()):
        return {"status": "unavailable", "message": "没有取得所需日期的有效天气数据，不可用当前数值替代。"}
    return {"status": "partial" if incomplete else "ok", "location": location,
            "local_time": local_now.isoformat(), "forecast_date": target.isoformat(), "period": query.period,
            "period_hours_local": {"day": "00:00-24:00", "morning": "06:00-12:00", "afternoon": "12:00-18:00", "evening": "18:00-24:00"}[query.period],
            "current_estimate": current_weather, "daily_forecast": selected_daily,
            "period_ranges": ranges, "weather_codes": sorted({row["weather_code"] for row in rows if row["weather_code"] is not None}),
            "weather_code_reference": "WMO: 0晴;1-3少云到阴;45/48雾;51-67毛毛雨/雨;71-77雪;80-82阵雨;85/86阵雪;95-99雷暴。没有雾霾预警字段。",
            "air_quality": air, "units": {"temperature": "°C", "humidity": "%", "wind": "m/s", "precipitation": "mm", "pm2_5": "µg/m³", "pm10": "µg/m³", "us_aqi": "US AQI"},
            "warnings": warnings, "sources": SOURCES}


def query_weather(query: WeatherQuery, *, client=None, now=None) -> dict:
    now = now or datetime.now(UTC)
    try:
        if client is None:
            with httpx.Client(timeout=4.0, follow_redirects=False) as http:
                result = _query(query, http, now)
        else:
            result = _query(query, client, now)
    except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError):
        result = {"status": "unavailable", "message": "天气服务暂不可用；请说明未查到，不要编造数值或预警。"}
    return {**result, "fetched_at": now.isoformat(), "provider": "Open-Meteo"}
