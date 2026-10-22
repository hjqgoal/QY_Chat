import json
from threading import Event
from unittest import TestCase
from unittest.mock import Mock, patch

from chat_reply_agent.config import Settings
from chat_reply_agent.schemas import ReplyControls, ChatAnalysis, ChatScenario, ExtractedChat
from chat_reply_agent.services import quality_queue_service as queue
from chat_reply_agent.services.llm_client import (
    ModelCallError, _quality_options, _request_timeout, call_text_model, model_request_timeout,
    model_role, quality_model_limits,
)
from chat_reply_agent.services.quality_repair_service import repair_candidates, repair_targets, requires_quality_gate
from chat_reply_agent.services.quality_service import review_candidates
from chat_reply_agent.services.telemetry_service import trace_scope
from fixtures_quality import reviewed_payload
from test_quality_queue import QueueHarness, TEXTS, review_input


class QualityWorkflowTests(QueueHarness, TestCase):
    def setUp(self):
        self.start_queue(workers=1)

    def review_response(self, defect=None):
        response = reviewed_payload(TEXTS)
        if defect == "low":
            response["items"][0]["review"]["naturalness"] = 3
        elif defect == "hard":
            response["items"][0]["review"]["hard_issues"] = ["未经确认的承诺"]
        elif defect == "unknown":
            response["items"][0]["review"]["scenario_fit"] = None
        return response

    def test_low_score_repairs_once_then_independently_rechecks(self):
        revised = ["好，那我先忙啦", *TEXTS[1:]]
        repair = {"items": [{"candidate_index": 0, "text": revised[0]}]}
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=[
                json.dumps(self.review_response("low")), json.dumps(reviewed_payload(revised))]) as judge, patch(
                "chat_reply_agent.services.quality_repair_service.call_text_model", return_value=json.dumps(repair)) as writer:
            job_id, _ = self.manager.submit("owner", review_input(), 90)
            state = self.terminal(job_id)
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.repair_status, "completed")
        self.assertEqual(state.candidate_texts, TEXTS)
        self.assertEqual(state.quality.items[0].evaluated_text, TEXTS[0])
        self.assertEqual([r.text for r in state.revision_replies], revised)
        self.assertEqual(state.revision_quality.items[0].evaluated_text, revised[0])
        self.assertTrue(state.revision_quality.passed)
        self.assertTrue(all(i.overall_score is None for i in state.revision_quality.items))
        self.assertEqual(judge.call_count, 2)
        writer.assert_called_once()
        self.assertEqual([s["stage"] for s in state.diagnostics["stages"]], ["quality.background", "quality.repair", "quality.recheck"])

    def test_recheck_failure_does_not_loop_or_publish_unreviewed_revision(self):
        for final in ("{}", ModelCallError("private reason", kind="request_timeout")):
            with self.subTest(final=type(final).__name__), patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=[
                    json.dumps(self.review_response("hard")), final]) as judge, patch(
                    "chat_reply_agent.services.quality_repair_service.call_text_model", return_value=json.dumps({
                        "items": [{"candidate_index": 0, "text": "好，那先这样"}]})) as writer:
                job_id, _ = self.manager.submit("owner", review_input(), 90)
                state = self.terminal(job_id)
                self.manager._jobs[job_id].future.result(2)
            self.assertEqual(state.status, "completed")
            self.assertIn(state.repair_status, {"failed", "timed_out"})
            self.assertFalse(state.quality.passed)
            self.assertEqual(state.revision_replies, [])
            self.assertEqual(state.diagnostics["phase"], "recheck")
            self.assertNotIn("private reason", state.model_dump_json())
            self.assertEqual(judge.call_count, 2)
            writer.assert_called_once()

    def test_unsatisfactory_recheck_is_shown_as_unverified_not_repaired_again(self):
        revised = ["好，那先这样", *TEXTS[1:]]
        recheck = reviewed_payload(revised)
        recheck["items"][0]["review"]["hard_issues"] = ["仍有问题"]
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=[json.dumps(self.review_response("low")), json.dumps(recheck)]) as judge, patch(
                "chat_reply_agent.services.quality_repair_service.call_text_model", return_value=json.dumps({"items": [{"candidate_index": 0, "text": revised[0]}]})) as writer:
            job_id, _ = self.manager.submit("owner", review_input(), 90)
            state = self.terminal(job_id)
        self.assertEqual(state.repair_status, "unverified")
        self.assertFalse(state.revision_quality.passed)
        self.assertEqual(judge.call_count, 2)
        writer.assert_called_once()

    def test_unknown_score_or_failed_review_never_triggers_repair(self):
        for raw in (json.dumps(self.review_response("unknown")), "{}", ModelCallError("secret", kind="rate_limit")):
            with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=raw if isinstance(raw, Exception) else None,
                       return_value=raw if isinstance(raw, str) else None), patch("chat_reply_agent.services.quality_repair_service.call_text_model") as writer:
                job_id, _ = self.manager.submit("owner", review_input(), 90)
                state = self.terminal(job_id)
                self.manager._jobs[job_id].future.result(2)
            writer.assert_not_called()
            self.assertEqual(state.repair_status, "not_needed")
            self.assertFalse(state.quality and state.quality.passed)

    def test_duplicate_active_or_expired_but_running_request_does_not_call_again(self):
        now = [0]
        self.manager._clock = lambda: now[0]
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=self.blocked_review) as judge:
            job_id, _ = self.manager.submit("owner", review_input(), 5)
            self.assertTrue(self.started.wait(1))
            self.assertEqual(self.manager.submit("owner", review_input(), 5), (job_id, "running"))
            now[0] = 6
            state = self.manager.get(job_id, "owner")
            self.assertEqual(state.status, "timed_out")
            self.assertEqual(state.diagnostics["phase"], "review")
            self.assertEqual(state.diagnostics["phase_elapsed_ms"], 6000)
            self.assertEqual(self.manager.submit("owner", review_input(), 5), (job_id, "timed_out"))
            self.release.set()
            self.manager._jobs[job_id].future.result(2)
        judge.assert_called_once()
        state = self.manager.get(job_id, "owner")
        self.assertEqual(state.status, "timed_out")
        self.assertTrue(state.diagnostics["stages"])
        self.assertFalse(state.diagnostics["worker_active"])
        self.assertIsNone(state.quality)

    def test_queue_wait_does_not_consume_model_budget(self):
        now, calls = [0], []
        second_started, second_release = Event(), Event()
        self.addCleanup(second_release.set)
        self.manager._clock = lambda: now[0]
        def reviewer(*args, **kwargs):
            calls.append(_request_timeout.get())
            if len(calls) == 1:
                return self.blocked_review(*args)
            second_started.set()
            second_release.wait(2)
            return json.dumps(reviewed_payload(TEXTS))
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=reviewer):
            first, _ = self.manager.submit("owner", review_input(), 5)
            self.assertTrue(self.started.wait(1))
            second, _ = self.manager.submit("owner", {**review_input(), "chat_text": "different"}, 5)
            now[0] = 4
            self.release.set()
            self.assertTrue(second_started.wait(1))
            self.assertEqual(calls, [5, 5])
            self.assertEqual(self.manager.get(second, "owner").diagnostics["queue_wait_ms"], 4000)
            now[0] = 8
            self.assertEqual(self.manager.get(second, "owner").status, "running")
            second_release.set()
            self.manager._jobs[first].future.result(2)
            self.manager._jobs[second].future.result(2)

    def test_cancel_during_repair_prevents_recheck_and_late_publication(self):
        def writer(*args, **kwargs):
            self.started.set()
            self.release.wait(2)
            return json.dumps({"items": [{"candidate_index": 0, "text": "好，那先这样"}]})
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=json.dumps(self.review_response("hard"))) as judge, patch(
                "chat_reply_agent.services.quality_repair_service.call_text_model", side_effect=writer):
            job_id, _ = self.manager.submit("owner", review_input(), 90)
            self.assertTrue(self.started.wait(1))
            self.assertEqual(self.manager.get(job_id, "owner").phase, "repair")
            self.manager.cancel(job_id, "owner")
            self.release.set()
            self.manager._jobs[job_id].future.result(2)
        judge.assert_called_once()
        self.assertEqual(self.manager.get(job_id, "owner").revision_replies, [])
        self.assertIsNone(self.manager.get(job_id, "owner").quality)

    def test_total_budget_ends_repair_without_publishing_or_rechecking(self):
        now = [0]
        self.manager._clock = lambda: now[0]
        settings = Settings(_env_file=None, quality_total_timeout_seconds=5)
        def writer(*args, **kwargs):
            now[0] = 6
            return json.dumps({"items": [{"candidate_index": 0, "text": "好，那先这样"}]})
        with patch.object(queue, "get_settings", return_value=settings), patch(
                "chat_reply_agent.services.quality_service.call_text_model", return_value=json.dumps(self.review_response("low"))) as judge, patch(
                "chat_reply_agent.services.quality_repair_service.call_text_model", side_effect=writer) as writer_mock:
            job_id, _ = self.manager.submit("owner", review_input(), 90)
            state = self.terminal(job_id)
        judge.assert_called_once()
        writer_mock.assert_called_once()
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.repair_status, "timed_out")
        self.assertEqual(state.diagnostics["error_code"], "workflow_timeout")
        self.assertIsNotNone(state.quality)
        self.assertFalse(state.revision_replies)


class QualityRepairContractTests(TestCase):
    def get_report(self):
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=json.dumps(reviewed_payload(TEXTS))):
            return review_candidates(**review_input(), targets={i: [] for i in range(3)}, audit_only=True).report

    def test_threshold_boundary_and_unknown_are_not_failure_defaults(self):
        report, replies = self.get_report(), review_input()["replies"]
        self.assertEqual(repair_targets(report, replies, 80), {})
        self.assertEqual(set(repair_targets(report, replies, 81)), {0, 1, 2})
        report.items[0].scenario_fit = None
        report.items[1].uncertainties = ["缺少必要背景"]
        report.items[2].hard_issues = ["凭空增加承诺"]
        self.assertEqual(set(repair_targets(report, replies, 81)), {2})
        replies[2].source = "fallback"
        self.assertEqual(repair_targets(report, replies, 81), {})

    def test_invalid_repair_targets_or_unchanged_text_are_rejected(self):
        report, payload = self.get_report(), review_input()
        cases = [[], [{"candidate_index": 1, "text": "不该改这条"}],
                 [{"candidate_index": 0, "text": TEXTS[0]}], [{"candidate_index": True, "text": "布尔值不是索引"}],
                 [{"candidate_index": 0, "text": ""}], [{"candidate_index": 0, "text": "a" * 601}],
                 [{"candidate_index": 0, "text": "新句"}, {"candidate_index": 0, "text": "重复索引"}]]
        for rows in cases:
            with self.subTest(rows=rows), patch("chat_reply_agent.services.quality_repair_service.call_text_model",
                    return_value=json.dumps({"items": rows})), self.assertRaises(ModelCallError) as error:
                repair_candidates(payload, report, {0: ["改第一条"]})
            self.assertEqual(error.exception.kind, "invalid_response")
            self.assertEqual([r.text for r in payload["replies"]], TEXTS)

    def test_review_requests_complete_schema_and_private_validation_diagnostics(self):
        broken = reviewed_payload(TEXTS)
        broken["items"][1]["review"]["evidence"]["relevance"] = " "
        broken["items"][2]["review"]["evaluated_text"] = "private chat that must not be logged"
        with trace_scope() as trace, patch("chat_reply_agent.services.quality_service.call_text_model",
                return_value=json.dumps(broken)) as model:
            result = review_candidates(**review_input(), targets={i: [] for i in range(3)}, audit_only=True)
        schema = model.call_args.kwargs["response_schema"]
        self.assertEqual(schema["properties"]["items"]["minItems"], 3)
        self.assertEqual(schema["properties"]["items"]["maxItems"], 3)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(result.report.status, "invalid_response")
        self.assertEqual(trace.validation_errors, [
            {"path": "items.1.review.evidence.relevance", "type": "missing_or_blank_evidence"},
            {"path": "items.2.review.evaluated_text", "type": "original_text_mismatch"}])
        self.assertNotIn("private chat", json.dumps(trace.snapshot()))


class QualityModelBudgetTests(TestCase):
    def test_provider_options_are_scoped_and_do_not_change_generation(self):
        settings = Settings(_env_file=None, openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", openai_model_name="qwen3.8-max")
        with patch("chat_reply_agent.services.llm_client.get_settings", return_value=settings):
            self.assertEqual(_quality_options(), {})
            with quality_model_limits(), model_role("review"):
                self.assertEqual(_quality_options(), {"max_completion_tokens": 6144, "extra_body": {"thinking_budget": 2048}})
                settings.openai_base_url = "https://other.example/v1"
                self.assertEqual(_quality_options(), {})
            self.assertEqual(_quality_options(), {})

    def test_review_timeout_overrides_general_timeout_and_usage_is_sanitized(self):
        settings = Settings(_env_file=None, openai_api_key="synthetic-test", model_timeout_seconds=10)
        client = Mock()
        client.chat.completions.create.return_value = Mock(choices=[Mock(finish_reason="stop", message=Mock(
            content="{}", tool_calls=None, refusal=None))], usage=Mock(prompt_tokens=100, completion_tokens=50,
            completion_tokens_details=Mock(reasoning_tokens=20)))
        with patch("chat_reply_agent.services.llm_client.get_settings", return_value=settings), patch(
                "chat_reply_agent.services.llm_client.OpenAI", return_value=client) as sdk, trace_scope() as trace:
            with quality_model_limits(), model_request_timeout(120), model_role("review"):
                call_text_model("private system", "private user")
        self.assertEqual(sdk.call_args.kwargs["timeout"], 120)
        self.assertEqual(trace.model_calls[0]["reasoning_tokens"], 20)
        self.assertNotIn("private", json.dumps(trace.snapshot()))

    def test_gate_includes_conflict_boundaries_and_decline(self):
        analysis, scenario, controls, chat = ChatAnalysis(risk_level="low"), ChatScenario(), ReplyControls(), ExtractedChat()
        self.assertFalse(requires_quality_gate(analysis, scenario, controls, chat))
        for value in ("conflict", "boundary", "apology"):
            self.assertTrue(requires_quality_gate(analysis, ChatScenario(type=value), controls, chat))
        self.assertTrue(requires_quality_gate(analysis, scenario, ReplyControls(intent="decline"), chat))
