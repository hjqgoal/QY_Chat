import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletionMessage

from chat_reply_agent.config import Settings
from chat_reply_agent.services.llm_client import (
    ModelCallError, call_text_model, model_role, reply_skill_tools, weather_tools,
)
from chat_reply_agent.services.reply_skill_service import (
    SKILL_IDS, _read_skill, load_reply_skills, reply_skill_catalog,
)
from chat_reply_agent.services.telemetry_service import trace_scope


def completion(calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls" if calls else "stop",
        message=ChatCompletionMessage(role="assistant", content=None if calls else '{"ok":true}', tool_calls=calls))])


def tool_call(name="load_reply_skills", arguments=None, call_id="s1"):
    return {"id": call_id, "type": "function", "function": {"name": name,
        "arguments": json.dumps(arguments if arguments is not None else {"skill_ids": ["clarify-facts"]})}}


class ReplySkillFilesTests(TestCase):
    def test_catalog_contains_metadata_only_and_all_resources_are_valid(self):
        catalog = reply_skill_catalog()
        self.assertEqual([item["id"] for item in catalog], list(SKILL_IDS))
        self.assertTrue(all(set(item) == {"id", "description"} for item in catalog))
        for skill_id in SKILL_IDS:
            meta, body = _read_skill(skill_id)
            self.assertEqual(meta["name"], skill_id)
            self.assertTrue(meta["metadata"]["version"])
            self.assertNotIn(body, json.dumps(catalog, ensure_ascii=False))

    def test_only_allowlisted_unique_ids_are_accepted_without_echoing_input(self):
        for arguments in (None, [], {}, {"skill_ids": []}, {"skill_ids": [True]},
                {"skill_ids": ["../../private"]}, {"skill_ids": ["clarify-facts"] * 2},
                {"skill_ids": list(SKILL_IDS)}, {"skill_ids": ["clarify-facts"], "chat": "PRIVATE"}):
            with self.subTest(arguments=arguments), patch(
                    "chat_reply_agent.services.reply_skill_service._read_skill") as read:
                result = load_reply_skills(arguments)
                self.assertEqual(result["status"], "invalid_arguments")
                self.assertNotIn("PRIVATE", json.dumps(result))
                read.assert_not_called()

    def test_success_logs_only_trusted_ids_and_versions(self):
        with trace_scope() as trace:
            result = load_reply_skills({"skill_ids": ["clarify-facts", "assert-position"]})
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["skills"]), 2)
            self.assertEqual(trace.reply_skills, [{"id": item["id"], "version": item["version"]}
                                                for item in result["skills"]])

    def test_missing_malformed_oversized_and_escaped_files_fail_closed(self):
        for failure in (FileNotFoundError(), ValueError("private details"), UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")):
            with patch("chat_reply_agent.services.reply_skill_service._read_skill", side_effect=failure):
                self.assertEqual(reply_skill_catalog(), [])
                self.assertEqual(load_reply_skills({"skill_ids": ["clarify-facts"]})["status"], "unavailable")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("chat_reply_agent.services.reply_skill_service.SKILL_ROOT", root):
                with patch.object(Path, "stat", return_value=SimpleNamespace(st_size=16001)):
                    with self.assertRaisesRegex(ValueError, "Oversized"):
                        _read_skill("clarify-facts")
                with patch.object(Path, "resolve", side_effect=[Path("/outside/SKILL.md"), root]):
                    with self.assertRaisesRegex(ValueError, "path"):
                        _read_skill("clarify-facts")
                with patch.object(Path, "stat", return_value=SimpleNamespace(st_size=100)), patch.object(
                        Path, "read_text", return_value="---\nname: wrong\n---\nbody"):
                    with self.assertRaisesRegex(ValueError, "skill"):
                        _read_skill("clarify-facts")


class ReplySkillLoopTests(TestCase):
    def invoke(self, responses, *, enabled=True, weather=False, role="strong"):
        settings = Settings(_env_file=None, reply_skills_enabled=enabled, weather_tool_enabled=weather,
                            model_structured_output="json_object")
        client, requests = Mock(), []
        def create(**kwargs):
            requests.append(deepcopy(kwargs))
            return responses[len(requests) - 1]
        client.chat.completions.create.side_effect = create
        with patch("chat_reply_agent.services.llm_client.get_settings", return_value=settings), patch(
                "chat_reply_agent.services.llm_client._client", return_value=client), trace_scope() as trace:
            with weather_tools() as evidence, reply_skill_tools(), model_role(role):
                output = call_text_model("base policy", "chat", response_schema={"type": "object"})
        client.close.assert_called_once()
        return output, requests, evidence, trace.snapshot()

    def test_simple_chat_finishes_in_one_round_with_metadata_not_bodies(self):
        with patch("chat_reply_agent.services.llm_client.load_reply_skills") as loader:
            _, requests, evidence, trace = self.invoke([completion()])
        self.assertEqual(len(requests), 1)
        self.assertIn("clarify-facts", requests[0]["messages"][0]["content"])
        self.assertNotIn(_read_skill("clarify-facts")[1], requests[0]["messages"][0]["content"])
        self.assertEqual(trace["reply_skills"], [])
        self.assertEqual(evidence, [])
        loader.assert_not_called()

    def test_selected_resources_arrive_only_after_tool_call_on_same_model(self):
        _, requests, evidence, trace = self.invoke([
            completion([tool_call(arguments={"skill_ids": ["clarify-facts", "assert-position"]})]), completion()])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["model"], requests[1]["model"])
        self.assertEqual(requests[0]["response_format"], requests[1]["response_format"])
        result = json.loads(requests[1]["messages"][-1]["content"])
        self.assertEqual([s["id"] for s in result["skills"]], ["clarify-facts", "assert-position"])
        self.assertEqual(requests[1]["tool_choice"], "none")
        self.assertEqual(len(trace["model_calls"]), 2)
        self.assertEqual(len(trace["reply_skills"]), 2)
        self.assertEqual(evidence, [])

    def test_weather_and_skills_share_three_round_budget_and_separate_evidence(self):
        weather = tool_call("get_weather", {"query": {"city": "北京"}}, "w1")
        weather2 = tool_call("get_weather", {"query": {"city": "上海"}}, "w2")
        with patch("chat_reply_agent.services.mcp_service.call_weather_tool", return_value={"status": "ok"}) as query:
            _, requests, evidence, _ = self.invoke([
                completion([tool_call(), weather]), completion([weather2]), completion()], weather=True)
        self.assertEqual(query.call_count, 2)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[-1]["tool_choice"], "none")
        self.assertEqual([item["tool"] for item in evidence], ["get_weather", "get_weather"])

    def test_second_skill_call_cannot_load_more_resources(self):
        _, requests, _, trace = self.invoke([completion([tool_call(), tool_call(call_id="s2")]), completion()])
        self.assertEqual(json.loads(requests[-1]["messages"][-1]["content"])["status"], "limit_reached")
        self.assertEqual(len(trace["reply_skills"]), 1)

    def test_invalid_or_unavailable_resource_returns_data_and_final_generation(self):
        for call, status in ((tool_call(arguments={"skill_ids": ["../private"]}), "invalid_arguments"),
                             (tool_call("read_any_file"), "unknown_tool")):
            _, requests, _, _ = self.invoke([completion([call]), completion()])
            self.assertEqual(json.loads(requests[-1]["messages"][-1]["content"])["status"], status)
        with patch("chat_reply_agent.services.reply_skill_service._read_skill", side_effect=OSError):
            _, requests, _, _ = self.invoke([completion([tool_call()]), completion()])
        self.assertEqual(json.loads(requests[-1]["messages"][-1]["content"])["status"], "unavailable")

    def test_disabled_review_and_memory_roles_never_receive_skill_tools(self):
        for options in ({"enabled": False}, {"role": "review"}, {"role": "memory"}):
            _, requests, _, _ = self.invoke([completion()], **options)
            self.assertNotIn("tools", requests[0])
            self.assertNotIn("可选策略目录", requests[0]["messages"][0]["content"])

    def test_bad_model_cannot_extend_loop_and_context_is_reset(self):
        from chat_reply_agent.services import llm_client
        with self.assertRaises(ModelCallError):
            self.invoke([completion([tool_call()])] * 3)
        self.assertFalse(llm_client._reply_skills.get())
        self.assertIsNone(llm_client._weather_results.get())
