"""Regression tests for ID-bound judgments and isolated partial failures."""

from copy import deepcopy
import json
from unittest import TestCase
from unittest.mock import patch

from fixtures_quality import DIMENSIONS, reviewed_payload, judge_response, judge_responses
from test_quality_queue import QueueHarness, review_input
from chat_reply_agent.services.quality_service import review_candidates
from chat_reply_agent.services.telemetry_service import trace_scope


class QualityBindingTests(TestCase):
    def run_review(self, mutate=lambda data: None, payload=None):
        payload = payload or review_input()
        def judge(*args, **kwargs):
            data = json.loads(judge_response(reviewed_payload([r.text for r in payload["replies"]]), *args, **kwargs))
            mutate(data)
            return json.dumps(data)
        with trace_scope() as trace, patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=judge) as model:
            result = review_candidates(**payload, targets={i: [] for i in range(3)}, audit_only=True, include_scores=False)
        return result, trace, model

    def test_quotes_newlines_and_reordering_bind_without_echoing_original(self):
        payload = review_input()
        payload["replies"][1].text = '你说“晚点聊”，那就晚点聊\n不用急，"等你"'
        result, _, model = self.run_review(lambda data: data["items"].reverse(), payload)
        self.assertEqual(result.replies, payload["replies"])
        self.assertEqual([item.evaluated_text for item in result.report.items], [r.text for r in payload["replies"]])
        self.assertEqual(result.report.status, "reviewed")
        schema = json.dumps(model.call_args.kwargs["response_schema"])
        for forbidden in ("evaluated_text", '"reply"', "candidate_index"):
            self.assertNotIn(forbidden, schema)
        self.assertIn("candidate_id", schema)
        self.assertTrue(all(item.overall_score is None for item in result.report.items))

    def test_missing_evidence_only_invalidates_affected_item(self):
        result, trace, _ = self.run_review(lambda data: data["items"][1]["review"]["evidence"].update(relevance=""))
        self.assertEqual([item.status for item in result.report.items], ["reviewed", "invalid_response", "reviewed"])
        self.assertEqual(result.report.status, "partial")
        self.assertFalse(result.report.passed)
        self.assertIsNone(result.report.average_score)
        self.assertEqual(trace.validation_errors, [{"path": "items.1.review.evidence.relevance", "type": "missing_or_blank_evidence"}])

    def test_duplicate_ids_invalidate_every_duplicate_not_first_or_last_wins(self):
        result, _, _ = self.run_review(lambda data: data["items"][1].update(candidate_id=data["items"][0]["candidate_id"]))
        self.assertEqual([item.status for item in result.report.items], ["invalid_response", "invalid_response", "reviewed"])

    def test_missing_unknown_and_unhashable_ids_are_not_inferred_from_position(self):
        mutations = [lambda data: data["items"].pop(1),
                     lambda data: data["items"][1].update(candidate_id="old-request:1"),
                     lambda data: data["items"][1].update(candidate_id=[1])]
        for mutate in mutations:
            result, _, _ = self.run_review(mutate)
            self.assertEqual(result.report.status, "partial")
            self.assertEqual(result.report.items[1].status, "invalid_response")
            self.assertEqual(result.report.items[2].status, "reviewed")

    def test_missing_group_evidence_keeps_single_items_but_not_group_approval(self):
        for mutate in (lambda data: data.pop("group_review"),
                       lambda data: data["group_review"].update(reason=""),
                       lambda data: data["group_review"].update(batch_id="stale")):
            result, _, _ = self.run_review(mutate)
            self.assertTrue(all(item.status == "reviewed" for item in result.report.items))
            self.assertEqual(result.report.status, "partial")
            self.assertFalse(result.report.passed)
            self.assertNotEqual(result.report.diversity_status, "distinct")

    def test_extra_rewrite_field_is_not_silently_applied_or_scored(self):
        result, _, _ = self.run_review(lambda data: data["items"][1].update(reply={"text": "不该改写"}))
        self.assertEqual(result.replies, review_input()["replies"])
        self.assertEqual(result.report.items[1].status, "invalid_response")
        self.assertEqual(result.report.items[0].status, "reviewed")

    def test_prior_request_ids_are_rejected_even_for_identical_text(self):
        saved = []
        self.run_review(lambda data: saved.append(deepcopy(data)))
        result, _, _ = self.run_review(lambda data: data.update(saved[0]))
        self.assertEqual(result.report.status, "invalid_response")
        self.assertTrue(all(not item.passed for item in result.report.items))

    def test_missing_score_is_not_defaulted_to_a_passing_value(self):
        for dimension in DIMENSIONS:
            result, _, _ = self.run_review(lambda data: data["items"][0]["review"].pop(dimension))
            self.assertEqual(result.report.items[0].status, "invalid_response")


class PartialQueueTests(QueueHarness, TestCase):
    def setUp(self):
        self.start_queue(workers=1)

    def test_partial_review_is_retained_and_never_starts_repair(self):
        def judge(*args, **kwargs):
            data = reviewed_payload([r.text for r in review_input()["replies"]])
            data["items"][1]["review"]["evidence"]["relevance"] = ""
            data["items"][0]["review"]["hard_issues"] = ["需要修改但本轮评价未完整完成"]
            return judge_response(data, *args, **kwargs)
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=judge) as model, patch(
                "chat_reply_agent.services.quality_repair_service.call_text_model") as writer:
            job, _ = self.manager.submit("owner", review_input(), 90)
            state = self.terminal(job)
        model.assert_called_once()
        writer.assert_not_called()
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.quality.status, "partial")
        self.assertFalse(state.quality.passed)
        self.assertIn("2/3", state.message)
        self.assertEqual(state.diagnostics["error_code"], "incomplete_evidence")
        self.assertEqual(state.revision_replies, [])

    def test_json_failure_has_distinct_diagnostic_code(self):
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value="{broken"):
            job, _ = self.manager.submit("owner", review_input(), 90)
            state = self.terminal(job)
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.diagnostics["error_code"], "invalid_json")

    def test_partial_recheck_keeps_original_review_and_does_not_publish_revision(self):
        texts = [r.text for r in review_input()["replies"]]
        original = reviewed_payload(texts)
        original["items"][0]["review"]["hard_issues"] = ["未经确认的承诺"]
        revised = ["好，那先这样", *texts[1:]]
        recheck = reviewed_payload(revised)
        recheck["items"][1]["review"]["evidence"]["relevance"] = ""
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=judge_responses(original, recheck)) as model, patch(
                "chat_reply_agent.services.quality_repair_service.call_text_model", return_value=json.dumps({
                    "items": [{"candidate_index": 0, "text": revised[0]}]})) as writer:
            job, _ = self.manager.submit("owner", review_input(), 90)
            state = self.terminal(job)
        self.assertEqual(model.call_count, 2)
        writer.assert_called_once()
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.repair_status, "failed")
        self.assertEqual(state.quality.items[0].evaluated_text, texts[0])
        self.assertEqual(state.revision_replies, [])
        self.assertEqual(state.diagnostics["error_code"], "incomplete_evidence")
