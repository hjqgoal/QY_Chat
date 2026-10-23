import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient
from openai import APITimeoutError, AuthenticationError
from pydantic import ValidationError

from fixtures_images import image_bytes
from chat_reply_agent.config import Settings
from chat_reply_agent.main import app
from chat_reply_agent.services.llm_client import (
    ModelCallError, _client, call_text_model, call_vision_model, model_request_timeout,
)
from chat_reply_agent.services.telemetry_service import trace_scope
from chat_reply_agent.services.vision_service import extract_chat_from_image, vision_json_schema, _parse_extracted_chat


VALID = json.dumps({"messages": [
    {"text": "差不多", "side": "right", "bubble_center_x": .8,
     "text_confidence": .95, "position_confidence": .95},
    {"text": "差不多", "side": "left", "bubble_center_x": .2,
     "text_confidence": .95, "position_confidence": .95},
]}, ensure_ascii=False)


class VisionRecoveryTests(TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None, openai_vision_model_name="qwen3.5-ocr",
                                 openai_vision_fallback_model_name="qwen3.8-max")
        self.override = patch("chat_reply_agent.services.vision_service.get_settings", return_value=self.settings)
        self.override.start()
        self.addCleanup(self.override.stop)

    def test_valid_primary_does_not_call_fallback(self):
        with patch("chat_reply_agent.services.vision_service.call_vision_model", return_value=VALID) as model:
            chat = extract_chat_from_image("synthetic.png")
        model.assert_called_once()
        self.assertEqual(model.call_args.kwargs["model_name"], "qwen3.5-ocr")
        self.assertFalse(chat.review_required)

    def test_compact_object_and_fenced_array_are_equivalent_without_recovery(self):
        rows = [{"side": "right", "text": "差不多"}, {"side": "left", "text": "差不多"}]
        for raw in (json.dumps({"messages": rows}), json.dumps(rows), "```json\n" + json.dumps(rows) + "\n```"):
            with patch("chat_reply_agent.services.vision_service.call_vision_model", return_value=raw) as model:
                chat = extract_chat_from_image("synthetic.png")
            model.assert_called_once()
            self.assertIsNone(model.call_args.kwargs["response_schema"])
            self.assertNotIn("bubble_center_x", model.call_args.args[1])
            self.assertEqual([m.speaker for m in chat.messages], ["me", "other"])
            self.assertEqual([m.text for m in chat.messages], ["差不多", "差不多"])
            self.assertTrue(chat.review_required)
            self.assertTrue(all(m.text_confidence <= .5 and m.speaker_confidence <= .5 for m in chat.messages))

    def test_invalid_array_prose_and_truncation_still_fail_closed(self):
        for raw in ('["普通OCR散文"]', '[{"text":"有文字但没有位置"}]', '[{"side":[],"text":"a"}]',
                    '[{"side":"right","text":"a"}', '说明：[{"side":"right","text":"a"}]'):
            with self.subTest(raw=raw), self.assertRaises(ModelCallError) as error:
                _parse_extracted_chat(raw)
            self.assertEqual(error.exception.kind, "invalid_structure")

    def test_invalid_empty_and_truncated_primary_recover_from_same_image_once(self):
        for failed in ("private-output-not-json", "", '{"messages": {}}', '{"messages": []}',
                       '{"messages": [{}]}', ModelCallError("truncated", kind="output_limit"),
                       ModelCallError("empty", kind="empty_response")):
            with self.subTest(failed=type(failed).__name__), trace_scope() as trace, patch(
                    "chat_reply_agent.services.vision_service.call_vision_model", side_effect=[failed, VALID]) as model:
                chat = extract_chat_from_image("synthetic.png")
            self.assertEqual(model.call_count, 2)
            self.assertEqual([c.args[0] for c in model.call_args_list], ["synthetic.png"] * 2)
            self.assertEqual([c.kwargs["model_name"] for c in model.call_args_list], ["qwen3.5-ocr", "qwen3.8-max"])
            self.assertEqual([m.speaker for m in chat.messages], ["me", "other"])
            self.assertEqual([m.text for m in chat.messages], ["差不多"] * 2)
            self.assertEqual(chat.last_message_from_other, "差不多")
            self.assertTrue(chat.review_required)
            self.assertEqual(chat.review_message_indexes, [0, 1])
            self.assertTrue(all(m.needs_review for m in chat.messages))
            self.assertTrue(all(m.speaker_source != "user_confirmed" for m in chat.messages))
            self.assertEqual(len(trace.validation_errors), 1)
            self.assertNotIn("private-output", json.dumps(trace.snapshot()))

    def test_two_invalid_outputs_stop_and_logs_do_not_contain_chat(self):
        with patch("chat_reply_agent.services.vision_service.call_vision_model", return_value="private-output-not-json") as model:
            with self.assertLogs("uvicorn.error", level="WARNING") as logs, self.assertRaises(ModelCallError) as error:
                extract_chat_from_image("private-filename.png")
        self.assertEqual(model.call_count, 2)
        self.assertEqual(error.exception.kind, "invalid_structure")
        self.assertIn("重新识别一次", str(error.exception))
        self.assertNotIn("private-", str(error.exception) + "\n".join(logs.output))

    def test_blank_fallback_retries_only_primary(self):
        self.settings.openai_vision_fallback_model_name = "  "
        with patch("chat_reply_agent.services.vision_service.call_vision_model", side_effect=["bad", VALID]) as model:
            extract_chat_from_image("synthetic.png")
        self.assertEqual([c.kwargs["model_name"] for c in model.call_args_list], ["qwen3.5-ocr"] * 2)

    def test_non_format_errors_are_not_retried(self):
        for kind in ("authentication_error", "permission_error", "model_not_found", "content_filter",
                     "rate_limit", "request_timeout", "connection_error", "model_error"):
            error = ModelCallError("safe-error", kind=kind)
            with self.subTest(kind=kind), patch("chat_reply_agent.services.vision_service.call_vision_model", side_effect=error) as model:
                with self.assertRaises(ModelCallError) as caught:
                    extract_chat_from_image("synthetic.png")
            self.assertEqual(caught.exception.kind, error.kind)
            self.assertIs(caught.exception.__cause__, error)
            self.assertIn("首次识别", str(caught.exception))
            model.assert_called_once()

    def test_recovery_provider_error_is_not_hidden_or_retried(self):
        with patch("chat_reply_agent.services.vision_service.call_vision_model", side_effect=[
                "bad", ModelCallError("no permission", kind="permission_error")]) as model:
            with self.assertRaises(ModelCallError) as error:
                extract_chat_from_image("synthetic.png")
        self.assertEqual(error.exception.kind, "permission_error")
        self.assertIn("自动重新识别", str(error.exception))
        self.assertIn("qwen3.8-max", str(error.exception))
        self.assertEqual(model.call_count, 2)

    def test_missing_configuration_keeps_placeholder_without_recovery(self):
        with patch("chat_reply_agent.services.vision_service.call_vision_model", return_value=None) as model:
            chat = extract_chat_from_image("synthetic.png")
        model.assert_called_once()
        self.assertEqual(chat.confidence, 0)
        self.assertTrue(chat.review_required)

    def test_status_exposes_loaded_vision_timeout_without_model_calls(self):
        with TestClient(app) as client, patch("chat_reply_agent.main.settings.vision_timeout_seconds", 150), patch(
                "chat_reply_agent.main.settings.openai_vision_fallback_model_name", "qwen3.8-max"), patch(
                "chat_reply_agent.services.llm_client.OpenAI") as sdk:
            response = client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["vision_timeout_seconds"], 150)
        self.assertEqual(response.json()["vision_fallback_model"], "qwen3.8-max")
        sdk.assert_not_called()

    def test_api_recovery_reaches_preview_and_cleans_up_success_and_failure(self):
        with TemporaryDirectory() as directory, TestClient(app) as client, patch(
                "chat_reply_agent.main.settings.upload_dir", directory):
            for route, field in (("/api/vision/extract", "image"), ("/api/vision/extract-batch", "images")):
                for second, status in ((VALID, 200), ("broken", 502)):
                    with self.subTest(route=route, status=status), patch(
                            "chat_reply_agent.services.vision_service.call_vision_model", side_effect=["broken", second]) as model, patch(
                            "chat_reply_agent.main._build_reply_response") as generate:
                        response = client.post(route, files={field: ("synthetic.png", image_bytes(), "image/png")})
                    self.assertEqual(response.status_code, status, response.text)
                    self.assertEqual(model.call_count, 2)
                    generate.assert_not_called()
                    self.assertEqual(list(Path(directory).iterdir()), [])
                    if status == 200:
                        self.assertTrue(response.json()["review_required"])
                        self.assertEqual(len(response.json()["messages"]), 2)


class VisionClientTests(TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None, openai_api_key="synthetic-key",
            openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_model_name="text-only-model", openai_vision_model_name="qwen3.5-ocr",
            openai_vision_fallback_model_name="qwen3.8-max", model_timeout_seconds=90,
            vision_timeout_seconds=180, model_structured_output="auto")
        self.override = patch("chat_reply_agent.services.llm_client.get_settings", return_value=self.settings)
        self.override.start()
        self.addCleanup(self.override.stop)

    def completion(self, content=VALID, *, finish="stop", refusal=None):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
            message=SimpleNamespace(content=content, refusal=refusal))])

    def invoke(self, response=None, model=None, error=None):
        client = Mock()
        client.chat.completions.create.return_value = response or self.completion()
        client.chat.completions.create.side_effect = error
        try:
            with patch("chat_reply_agent.services.llm_client._client", return_value=client) as factory, patch(
                    "chat_reply_agent.services.llm_client.Path.read_bytes", return_value=image_bytes()):
                output = call_vision_model("synthetic.png", "return JSON", model_name=model, response_schema=vision_json_schema())
            return output, client.chat.completions.create.call_args.kwargs
        finally:
            factory.assert_called_once_with(timeout_seconds=self.settings.vision_timeout_seconds)
            client.close.assert_called_once()

    def test_ocr_does_not_receive_unsupported_structured_parameters(self):
        for mode in ("auto", "json_schema", "json_object", "prompt"):
            self.settings.model_structured_output = mode
            output, request = self.invoke()
            self.assertEqual(output, VALID)
            self.assertEqual(request["model"], "qwen3.5-ocr")
            self.assertNotIn("response_format", request)

    def test_fallback_schema_uses_vision_model_not_text_role(self):
        _, request = self.invoke(model="qwen3.8-max")
        self.assertEqual(request["model"], "qwen3.8-max")
        fmt = request["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["json_schema"]["name"], "chat_extraction")
        self.assertTrue(fmt["json_schema"]["strict"])
        self.assertEqual(fmt["json_schema"]["schema"], vision_json_schema())
        self.assertTrue(request["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_empty_truncated_filtered_refused_outputs_have_safe_error_kinds(self):
        for response, kind in ((SimpleNamespace(choices=[]), "empty_response"),
                               (self.completion(None), "empty_response"),
                               (self.completion(" "), "empty_response"),
                               (self.completion(finish="length"), "output_limit"),
                               (self.completion(finish="content_filter"), "content_filter"),
                               (self.completion(refusal="private-refusal", finish="length"), "content_filter")):
            with self.subTest(kind=kind), self.assertRaises(ModelCallError) as error:
                self.invoke(response)
            self.assertEqual(error.exception.kind, kind)
            self.assertNotIn("private-refusal", str(error.exception))

    def test_transport_error_preserves_safe_kind_and_closes_client(self):
        failure = AuthenticationError("private-response", response=httpx.Response(401,
            request=httpx.Request("POST", "https://example.invalid")), body=None)
        with self.assertRaises(ModelCallError) as error:
            self.invoke(error=failure)
        self.assertEqual(error.exception.kind, "authentication_error")
        self.assertNotIn("private-response", str(error.exception))

    def test_vision_timeout_is_independent_and_text_timeout_is_unchanged(self):
        with patch("chat_reply_agent.services.llm_client.OpenAI") as sdk:
            _client(timeout_seconds=self.settings.vision_timeout_seconds)
            self.assertEqual(sdk.call_args.kwargs["timeout"], 180)
            self.assertEqual(sdk.call_args.kwargs["max_retries"], 0)
            sdk.return_value.chat.completions.create.return_value = self.completion()
            call_text_model("system", "synthetic input")
            self.assertEqual(sdk.call_args.kwargs["timeout"], 90)

    def test_vision_timeout_respects_smaller_outer_budget(self):
        with patch("chat_reply_agent.services.llm_client.OpenAI") as sdk:
            with model_request_timeout(12):
                _client(timeout_seconds=180)
                self.assertEqual(sdk.call_args.kwargs["timeout"], 12)
            with model_request_timeout(250):
                _client(timeout_seconds=180)
                self.assertEqual(sdk.call_args.kwargs["timeout"], 180)
                _client()
                self.assertEqual(sdk.call_args.kwargs["timeout"], 250)

    def test_sdk_timeout_has_specific_safe_message(self):
        failure = APITimeoutError(request=httpx.Request("POST", "https://example.invalid/private-path"))
        with self.assertRaises(ModelCallError) as error:
            self.invoke(error=failure)
        self.assertEqual(error.exception.kind, "request_timeout")
        self.assertIn("180 秒", str(error.exception))
        self.assertIn("不会继续自动重试", str(error.exception))
        self.assertNotIn("OPENAI_API_KEY", str(error.exception))
        self.assertNotIn("private-path", str(error.exception))

    def test_api_timeout_identifies_primary_or_fallback_and_cleans_upload(self):
        with TemporaryDirectory() as directory, TestClient(app) as api, patch(
                "chat_reply_agent.main.settings.upload_dir", directory), patch(
                "chat_reply_agent.services.vision_service.get_settings", return_value=self.settings):
            for attempt, name in ((1, "qwen3.5-ocr"), (2, "qwen3.8-max")):
                client = Mock()
                failure = APITimeoutError(request=httpx.Request("POST", "https://example.invalid/private-path"))
                client.chat.completions.create.side_effect = [self.completion("broken")] * (attempt - 1) + [failure]
                with patch("chat_reply_agent.services.llm_client.OpenAI", return_value=client), self.assertLogs(
                        "uvicorn.error", level="WARNING") as logs:
                    response = api.post("/api/vision/extract-batch", files={
                        "images": ("private-filename.png", image_bytes(), "image/png")})
                self.assertEqual(response.status_code, 502, response.text)
                self.assertIn("首次识别" if attempt == 1 else "自动重新识别", response.text)
                self.assertIn(name, response.text)
                self.assertIn("180 秒", response.text)
                self.assertEqual(client.chat.completions.create.call_count, attempt)
                self.assertEqual(client.close.call_count, attempt)
                self.assertEqual(list(Path(directory).iterdir()), [])
                logged = "\n".join(logs.output)
                self.assertIn(f"attempt={attempt} model={name} type=request_timeout", logged)
                self.assertIn("duration_ms=", logged)
                self.assertNotIn("private-", logged + response.text)
                self.assertNotIn("synthetic-key", logged + response.text)

    def test_vision_timeout_configuration_is_bounded(self):
        for timeout in (0, -1, 301, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValidationError):
                Settings(_env_file=None, vision_timeout_seconds=timeout)
