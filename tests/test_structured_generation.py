import json
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletionMessage

from fixtures_v021 import GENERATION
from chat_reply_agent.config import Settings
from chat_reply_agent.services.generation_service import generation_json_schema, understand_and_generate
from chat_reply_agent.services.llm_client import ModelCallError, call_text_model, weather_tools
from chat_reply_agent.services.scenario_service import load_scenario_rules
from chat_reply_agent.services.skill_loader import load_relation_skill
from chat_reply_agent.services.telemetry_service import trace_scope


class StructuredGenerationTests(TestCase):
    def setUp(self):
        self.schema = generation_json_schema(load_scenario_rules())
        self.settings = Settings(_env_file=None, openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                 openai_model_name="qwen3.8-max", openai_strong_model_name="",
                                 model_structured_output="auto")

    def completion(self, content=GENERATION, *, finish_reason="stop", calls=None, refusal=None):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
            message=ChatCompletionMessage(role="assistant", content=content, tool_calls=calls, refusal=refusal))])

    def invoke(self, responses=None, schema=True):
        client = Mock()
        requests = []
        responses = responses or [self.completion()]
        def create(**kwargs):
            requests.append(deepcopy(kwargs))
            return responses[len(requests) - 1]
        client.chat.completions.create.side_effect = create
        with patch("chat_reply_agent.services.llm_client.get_settings", return_value=self.settings), patch(
                "chat_reply_agent.services.llm_client._client", return_value=client):
            output = call_text_model("system", "return JSON", response_schema=self.schema if schema else None)
        client.close.assert_called_once()
        return output, requests

    def test_supported_qwen_receives_strict_schema_and_scenario_enum(self):
        _, requests = self.invoke()
        self.assertEqual(len(requests), 1)
        fmt = requests[0]["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertTrue(fmt["json_schema"]["strict"])
        schema = fmt["json_schema"]["schema"]
        self.assertEqual(set(schema["required"]), {"topic", "analysis", "scenario", "strategy", "replies"})
        self.assertEqual(schema["properties"]["replies"]["minItems"], 3)
        self.assertEqual(schema["properties"]["replies"]["maxItems"], 3)
        self.assertEqual(schema["$defs"]["ScenarioChoice"]["properties"]["type"]["enum"], list(load_scenario_rules()))
        self.assertTrue(all(not node["additionalProperties"] for node in [schema, *schema["$defs"].values()]))

    def test_snapshot_support_and_provider_boundary(self):
        self.settings.openai_model_name = "qwen3.8-max-0902"
        self.assertEqual(self.invoke()[1][0]["response_format"]["type"], "json_schema")
        self.settings.openai_base_url = "https://example.invalid/compatible-mode/v1"
        self.assertEqual(self.invoke()[1][0]["response_format"], {"type": "json_object"})
        self.settings.openai_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.settings.openai_model_name = "qwen3.6-flash"
        self.assertEqual(self.invoke()[1][0]["response_format"], {"type": "json_object"})

    def test_overrides_and_unstructured_calls_preserve_compatibility(self):
        for mode in ("json_schema", "json_object", "prompt"):
            self.settings.model_structured_output = mode
            request = self.invoke()[1][0]
            if mode == "prompt":
                self.assertNotIn("response_format", request)
            else:
                self.assertEqual(request["response_format"]["type"], mode)
            self.assertNotIn("response_format", self.invoke(schema=False)[1][0])

    def test_tool_round_trip_keeps_schema_and_evidence(self):
        call = {"id": "weather-1", "type": "function", "function": {"name": "get_weather",
                "arguments": json.dumps({"query": {"city": "北京"}})}}
        with patch("chat_reply_agent.services.mcp_service.call_weather_tool", return_value={"status": "ok"}) as tool:
            with weather_tools() as evidence:
                output, requests = self.invoke([
                    self.completion(None, calls=[call], finish_reason="tool_calls"), self.completion()])
        self.assertEqual(output, GENERATION)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["response_format"], requests[1]["response_format"])
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(evidence[0]["result"]["status"], "ok")
        tool.assert_called_once()

    def test_truncated_filtered_refused_and_empty_results_fail_before_parsing(self):
        for completion, expected in (
            (self.completion(finish_reason="length"), "截断"),
            (self.completion(finish_reason="content_filter"), "未能处理"),
            (self.completion(refusal="private refusal"), "未能处理"),
            (self.completion("  "), "空内容"),
            (SimpleNamespace(choices=[]), "没有返回"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(ModelCallError, expected):
                self.invoke([completion])

    def generate(self, raw):
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=raw) as model:
            result = understand_and_generate("对方：刚忙完。", load_relation_skill("friend"), "", "")
        return result, model

    def test_pipeline_passes_schema_and_still_accepts_valid_json_fences(self):
        result, model = self.generate("```json\n" + GENERATION + "\n```")
        self.assertEqual(len(result.replies), 3)
        self.assertEqual(model.call_args.kwargs["response_schema"], self.schema)
        model.assert_called_once()

    def test_broken_json_is_rejected_without_guessing_or_logging_private_text(self):
        raw = '```json\n{"strategy":"private-chat\n</parameter>\n"replies":[]}\n```'
        with trace_scope() as trace, self.assertLogs("uvicorn.error", level="WARNING") as logs:
            with self.assertRaisesRegex(ModelCallError, "JSON"):
                self.generate(raw)
        output = "\n".join(logs.output)
        self.assertIn("invalid_or_empty_json_object", output)
        self.assertIn(trace.request_id, output)
        self.assertNotIn("private-chat", output)
        self.assertNotIn("</parameter>", output)

    def test_schema_diagnostics_only_log_field_paths_and_error_types(self):
        payload = json.loads(GENERATION)
        payload["scenario"]["confidence"] = "private-chat"
        with self.assertLogs("uvicorn.error", level="WARNING") as logs, self.assertRaises(ModelCallError):
            self.generate(json.dumps(payload))
        output = "\n".join(logs.output)
        self.assertIn("scenario.confidence", output)
        self.assertIn("float_type", output)
        self.assertNotIn("private-chat", output)

