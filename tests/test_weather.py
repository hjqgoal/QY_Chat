import asyncio
from copy import deepcopy
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch

import httpx
from openai.types.chat import ChatCompletionMessage

from chat_reply_agent.config import Settings
from chat_reply_agent.services.llm_client import call_text_model, weather_tools, model_role, ModelCallError
from chat_reply_agent.services.mcp_service import call_weather_tool, _cache, _call_mcp
from chat_reply_agent.services.weather_contract import WeatherQuery
from chat_reply_agent.services.weather_service import query_weather


NOW = datetime(2026, 10, 9, 0, tzinfo=UTC)
PLACE = {"name": "北京", "country_code": "CN", "country": "中国", "admin1": "北京", "latitude": 39.9, "longitude": 116.4, "timezone": "Asia/Shanghai"}


def forecast():
    from chat_reply_agent.services.weather_service import DAILY_FIELDS, HOURLY_FIELDS
    times = [f"2026-10-{day:02d}T{hour:02d}:00" for day in (9, 10) for hour in range(24)]
    hourly = {key: [10.0] * 48 for key in HOURLY_FIELDS}
    hourly["relative_humidity_2m"] = [60.0] * 48
    hourly["weather_code"] = [3] * 48
    return {"current": {"time": "2026-10-09T08:00", "temperature_2m": 99},
            "daily": {"time": ["2026-10-09", "2026-10-10"], **{key: [12, 13] for key in DAILY_FIELDS}},
            "hourly": {"time": times, **hourly}}


class WeatherServiceTests(TestCase):
    def query(self, query=None, locations=None, air_fail=False, payload=None, air_null=False):
        calls = []
        def handler(request):
            calls.append(request)
            if "geocoding" in request.url.host:
                data = {"results": locations if locations is not None else [PLACE]}
            elif "air-quality" in request.url.host:
                if air_fail:
                    return httpx.Response(503)
                data = {"hourly": {"time": forecast()["hourly"]["time"], **{
                    key: [None if air_null else 25] * 48 for key in ("pm2_5", "pm10", "us_aqi")}}}
            else:
                data = payload if payload is not None else forecast()
            return httpx.Response(200, json=data)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = query_weather(query or WeatherQuery(city="北京"), client=client, now=NOW)
        return result, calls

    def test_weather_units_and_air_standard_are_explicit(self):
        result, calls = self.query()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["units"]["wind"], "m/s")
        self.assertEqual(result["period_ranges"]["relative_humidity_2m"]["max"], 60)
        self.assertEqual(result["air_quality"]["standard"], "US AQI")
        self.assertEqual(result["forecast_date"], "2026-10-09")
        self.assertEqual(result["fetched_at"], NOW.isoformat())
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1].url.params["wind_speed_unit"], "ms")
        self.assertNotIn("api_key", str(calls))

    def test_tomorrow_never_substitutes_current_weather(self):
        result, _ = self.query(WeatherQuery(city="北京", day="tomorrow", period="evening"))
        self.assertEqual(result["forecast_date"], "2026-10-10")
        self.assertIsNone(result["current_estimate"])
        self.assertEqual(result["period_ranges"]["temperature_2m"], {"min": 10, "max": 10, "available_hours": 6})
        self.assertEqual(result["period_hours_local"], "18:00-24:00")

    def test_local_date_uses_city_timezone(self):
        place = {**PLACE, "timezone": "America/Los_Angeles"}
        result, _ = self.query(WeatherQuery(city="北京", day="2026-10-15"), locations=[place])
        self.assertEqual(result["status"], "unsupported_date")
        self.assertTrue(result["local_time"].startswith("2026-10-08"))

    def test_ambiguous_and_unknown_cities_do_not_fetch_forecasts(self):
        result, calls = self.query(locations=[PLACE, {**PLACE, "admin1": "同名地区"}])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(len(calls), 1)
        result, calls = self.query(locations=[])
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(len(calls), 1)
        result, _ = self.query(WeatherQuery(city="北京", region="北京"), locations=[PLACE, {**PLACE, "admin1": "同名地区"}])
        self.assertEqual(result["status"], "ok")

    def test_history_and_out_of_range_dates_are_rejected(self):
        for day in ("2026-10-08", "2026-10-16"):
            result, calls = self.query(WeatherQuery(city="北京", day=day))
            self.assertEqual(result["status"], "unsupported_date")
            self.assertEqual(len(calls), 1)

    def test_air_failure_or_null_is_not_reported_as_clean_air(self):
        result, _ = self.query(air_fail=True)
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["air_quality"])
        self.assertIsNotNone(result["period_ranges"]["temperature_2m"])
        result, _ = self.query(air_null=True)
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["air_quality"]["period_ranges"]["pm2_5"])

    def test_missing_or_invalid_forecast_is_unavailable(self):
        result, _ = self.query(payload={})
        self.assertEqual(result["status"], "unavailable")

    def test_partial_hourly_coverage_is_not_labelled_complete(self):
        payload = forecast()
        payload["hourly"]["temperature_2m"][0] = None
        result, _ = self.query(payload=payload)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["period_ranges"]["temperature_2m"]["available_hours"], 23)
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(429))) as client:
            result = query_weather(WeatherQuery(city="北京"), client=client, now=NOW)
        self.assertEqual(result["status"], "unavailable")

    def test_parameters_reject_urls_and_unrelated_data(self):
        for value in ({"city": "http://localhost"}, {"city": "北京", "day": "yesterday"},
                      {"city": "北京", "chat": "private"}, {"city": "北京", "period": "invalid"}):
            with self.assertRaises(ValueError):
                WeatherQuery.model_validate(value)


class ToolLoopTests(TestCase):
    def completion(self, content=None, calls=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=ChatCompletionMessage(
            role="assistant", content=content, tool_calls=calls))])

    def tool_call(self, name="get_weather", arguments=None, call_id="weather-1"):
        return {"id": call_id, "type": "function", "function": {"name": name,
            "arguments": arguments if arguments is not None else json.dumps({"query": {"city": "北京"}})}}

    def run_model(self, responses, result=None, enabled=True):
        client = Mock()
        requests = []
        def create(**kwargs):
            requests.append(deepcopy(kwargs))
            return responses[len(requests) - 1]
        client.chat.completions.create.side_effect = create
        settings = Settings(_env_file=None, weather_tool_enabled=enabled)
        with patch("chat_reply_agent.services.llm_client._client", return_value=client), patch(
                "chat_reply_agent.services.llm_client.get_settings", return_value=settings), patch(
                "chat_reply_agent.services.mcp_service.call_weather_tool", return_value=result or {"status": "ok", "temperature": 14}) as tool:
            with weather_tools() as evidence:
                output = call_text_model("system", "user")
        return output, evidence, requests, tool

    def test_ordinary_chat_is_one_call_and_starts_no_mcp(self):
        output, evidence, requests, tool = self.run_model([self.completion("plain")])
        self.assertEqual(output, "plain")
        self.assertEqual(evidence, [])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["tool_choice"], "auto")
        tool.assert_not_called()

    def test_tool_result_is_sent_back_to_same_model_before_final_response(self):
        output, evidence, requests, tool = self.run_model([
            self.completion(calls=[self.tool_call()]), self.completion("final")])
        self.assertEqual(output, "final")
        tool.assert_called_once_with({"query": {"city": "北京"}})
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "weather-1")
        self.assertEqual(json.loads(requests[1]["messages"][-1]["content"])["temperature"], 14)
        self.assertEqual(evidence[0]["result"]["status"], "ok")

    def test_tool_failure_is_evidence_not_a_success_claim(self):
        _, evidence, requests, _ = self.run_model([self.completion(calls=[self.tool_call()]), self.completion("ask")],
                                                 result={"status": "unavailable"})
        self.assertEqual(evidence[0]["result"]["status"], "unavailable")
        self.assertIn("unavailable", requests[1]["messages"][-1]["content"])

    def test_unknown_tools_and_malformed_json_are_not_executed(self):
        for call in (self.tool_call(name="run_shell"), self.tool_call(arguments="not json")):
            _, _, _, tool = self.run_model([self.completion(calls=[call]), self.completion("safe")])
            tool.assert_not_called()

    def test_tool_budget_is_two_executions_and_final_round_disables_tools(self):
        _, _, requests, tool = self.run_model([
            self.completion(calls=[self.tool_call()]),
            self.completion(calls=[self.tool_call(call_id="weather-2")]), self.completion("final")])
        self.assertEqual(tool.call_count, 2)
        self.assertEqual(requests[-1]["tool_choice"], "none")

    def test_model_cannot_keep_calling_after_budget(self):
        with self.assertRaises(ModelCallError):
            self.run_model([self.completion(calls=[self.tool_call()])] * 3)

    def test_disabled_tool_adds_no_tools_argument(self):
        _, _, requests, tool = self.run_model([self.completion("safe")], enabled=False)
        self.assertNotIn("tools", requests[0])
        tool.assert_not_called()

    def test_tool_collection_is_reset_after_request(self):
        from chat_reply_agent.services.llm_client import _weather_results
        self.run_model([self.completion(calls=[self.tool_call()]), self.completion("safe")])
        self.assertIsNone(_weather_results.get())

    def test_review_and_memory_roles_do_not_get_weather_tools(self):
        for role in ("review", "memory"):
            with model_role(role):
                _, _, requests, tool = self.run_model([self.completion("safe")])
            self.assertNotIn("tools", requests[0])
            tool.assert_not_called()


class MCPBridgeTests(TestCase):
    def setUp(self):
        _cache.clear()
        self.settings = Settings(_env_file=None)
        self.setting_patch = patch("chat_reply_agent.services.mcp_service.get_settings", return_value=self.settings)
        self.setting_patch.start()
        self.addCleanup(self.setting_patch.stop)

    def test_invalid_arguments_fail_before_starting_mcp(self):
        with patch("chat_reply_agent.services.mcp_service._call_mcp") as backend:
            for value in ({"query": {"city": "北京", "secret": "private"}}, {"url": "http://localhost"}, []):
                self.assertEqual(call_weather_tool(value)["status"], "invalid_arguments")
        backend.assert_not_called()

    def test_cache_preserves_fetch_time_and_does_not_cache_failure(self):
        success = {"status": "ok", "location": {"timezone": "Asia/Shanghai"}, "fetched_at": datetime.now(UTC).isoformat()}
        with patch("chat_reply_agent.services.mcp_service._call_mcp", return_value=success) as backend:
            first = call_weather_tool({"query": {"city": "北京"}})
            second = call_weather_tool({"query": {"city": "北京"}})
        backend.assert_called_once()
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["fetched_at"], second["fetched_at"])
        _cache.clear()
        with patch("chat_reply_agent.services.mcp_service._call_mcp", return_value={"status": "unavailable"}) as backend:
            call_weather_tool({"query": {"city": "北京"}})
            call_weather_tool({"query": {"city": "北京"}})
        self.assertEqual(backend.call_count, 2)

    def test_cache_expires_at_city_midnight(self):
        success = {"status": "ok", "location": {"timezone": "Asia/Shanghai"}, "fetched_at": "2000-01-01T00:00:00+00:00"}
        with patch("chat_reply_agent.services.mcp_service._call_mcp", return_value=success) as backend:
            call_weather_tool({"query": {"city": "北京"}})
            call_weather_tool({"query": {"city": "北京"}})
        self.assertEqual(backend.call_count, 2)

    @skipUnless(importlib.util.find_spec("mcp"), "Install MCP dependencies for protocol integration tests")
    def test_real_stdio_roundtrip_and_no_model_credentials_in_child(self):
        from mcp import StdioServerParameters
        def parameters(**kwargs):
            self.assertEqual(kwargs["args"], ["-m", "chat_reply_agent.mcp_server"])
            kwargs["args"] = [str(Path(__file__).with_name("mcp_weather_fixture.py"))]
            return StdioServerParameters(**kwargs)
        with patch("mcp.StdioServerParameters", side_effect=parameters), patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret"}):
            result = asyncio.run(_call_mcp({"query": {"city": "北京", "day": "tomorrow"}}))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["day"], "tomorrow")
        self.assertFalse(result["has_model_secret"])

    @skipUnless(importlib.util.find_spec("mcp"), "Install MCP dependencies for protocol integration tests")
    def test_stdio_timeout_is_bounded_and_reported_as_failure(self):
        from mcp import StdioServerParameters
        self.settings.weather_tool_timeout_seconds = 3
        def parameters(**kwargs):
            kwargs["args"] = [str(Path(__file__).with_name("mcp_weather_fixture.py"))]
            return StdioServerParameters(**kwargs)
        with patch("mcp.StdioServerParameters", side_effect=parameters):
            result = call_weather_tool({"query": {"city": "Timeout"}})
        self.assertEqual(result["status"], "unavailable")


class WeatherPipelineTests(TestCase):
    def setUp(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from chat_reply_agent.database import Base
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self.evidence = {"tool": "get_weather", "result": {"status": "ok", "provider": "test-weather-provider"}}

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def emit_result(self, text):
        from chat_reply_agent.services.llm_client import _weather_results
        _weather_results.get().append(self.evidence)
        return text

    def test_weather_evidence_reaches_review_and_ui_but_not_memory_payload(self):
        from fixtures_v021 import GENERATION
        from chat_reply_agent.main import _build_reply_response
        from chat_reply_agent.models import MemoryJob
        from chat_reply_agent.schemas import ExtractedChat
        from chat_reply_agent.services.directed_review_service import review_targeted_replies
        with patch("chat_reply_agent.services.generation_service.call_text_model", side_effect=lambda *args, **kwargs: self.emit_result(GENERATION)), patch(
                "chat_reply_agent.main.review_targeted_replies", wraps=review_targeted_replies) as reviewer:
            response = _build_reply_response(self.db, "weather-user", "friend", "friend", "朋友",
                                             ExtractedChat(), background_context="对方在北京，问明天穿什么。")
        self.assertEqual(response.weather_evidence, [self.evidence])
        self.assertIn("test-weather-provider", reviewer.call_args.kwargs["chat_text"])
        payload = self.db.get(MemoryJob, response.memory_job_id).payload_json
        self.assertNotIn("test-weather-provider", payload)
        self.assertNotIn("weather_evidence", payload)

    def test_single_regeneration_collects_new_weather_and_supplies_reviewer(self):
        from fixtures_v021 import GENERATION
        from chat_reply_agent.main import regenerate_single_reply
        from chat_reply_agent.schemas import RegenerateReplyRequest
        from chat_reply_agent.services.directed_review_service import review_targeted_replies
        data = json.loads(GENERATION)
        request = RegenerateReplyRequest(relation_type="friend", extracted_chat={"messages": []},
            background_context="对方在北京，问明天穿什么。", analysis=data["analysis"], scenario=data["scenario"],
            existing_replies=data["replies"], candidate_index=0)
        answer = json.dumps({"reply": {"text": "带件外套吧", "style": "自然", "reason": "结合天气"}})
        with patch("chat_reply_agent.services.reply_service.call_text_model", side_effect=lambda *args: self.emit_result(answer)), patch(
                "chat_reply_agent.main.review_targeted_replies", wraps=review_targeted_replies) as reviewer:
            response = regenerate_single_reply(request, self.db)
        self.assertEqual(response.weather_evidence, [self.evidence])
        self.assertIn("test-weather-provider", reviewer.call_args.kwargs["chat_text"])
