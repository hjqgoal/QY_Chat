"""Deterministic report semantics; mocked judgments are not model benchmarks."""

from unittest import TestCase
from unittest.mock import patch

from chat_reply_agent.config import Settings
from chat_reply_agent.schemas import ChatAnalysis, ChatMessage, ChatScenario, ExtractedChat, ReplyControls
from chat_reply_agent.services.llm_client import _quality_options, quality_model_limits, model_role
from chat_reply_agent.services.quality_queue_service import select_review_mode
from fixtures_quality import reviewed_payload, judge_responses
import test_quality_binding as binding_tests
from test_quality_queue import QueueHarness, review_input


class UsefulReportTests(TestCase):
    run_review = binding_tests.QualityBindingTests.run_review

    def test_ordinary_advice_and_missing_style_do_not_block(self):
        def mutate(data):
            for row in data["items"]:
                row["review"]["advisories"] = ["无法保证对方反应", "缺少本人风格样本"]
        result, _, _ = self.run_review(mutate)
        report = result.report
        self.assertTrue(report.passed)
        self.assertEqual(report.recommended_candidate_indexes, [0, 1, 2])
        self.assertIn("唯一最佳", report.recommendation_reason)
        self.assertIn("3 条基本可用", report.summary)
        self.assertTrue(report.group_reason)
        for item in report.items:
            self.assertEqual(item.verdict, "usable")
            self.assertEqual(item.uncertainties, [])
            self.assertEqual(len(item.advisories), 2)
            self.assertIsNone(item.style_fit)
            self.assertIsNone(item.overall_score)
            self.assertTrue(item.conclusion)

    def test_critical_unknown_blocks_only_affected_candidate(self):
        result, _, _ = self.run_review(lambda data: data["items"][1]["review"].update(
            critical_unknowns=["尚未确认能否明天赴约，会影响这条承诺是否成立"]))
        self.assertEqual(result.report.items[1].verdict, "uncertain")
        self.assertIn("赴约", result.report.items[1].conclusion)
        self.assertEqual(result.report.recommended_candidate_indexes, [0, 2])
        self.assertIn("1 条需补充关键信息", result.report.summary)
        self.assertFalse(result.report.passed)

    def test_null_core_still_blocks_even_with_empty_unknowns(self):
        result, _, _ = self.run_review(lambda data: data["items"][0]["review"].update(relevance=None))
        self.assertEqual(result.report.items[0].verdict, "uncertain")
        self.assertNotIn(0, result.report.recommended_candidate_indexes)

    def test_missing_or_invalid_unknown_categories_fail_closed(self):
        for key in ("critical_unknowns", "advisories"):
            for mutate in (lambda data: data["items"][0]["review"].pop(key),
                           lambda data: data["items"][0]["review"].update({key: [None]}),
                           lambda data: data["items"][0]["review"].update({key: [" "]})):
                result, _, _ = self.run_review(mutate)
                self.assertEqual(result.report.items[0].status, "invalid_response")
                self.assertEqual(result.report.recommended_candidate_indexes, [])

    def test_hard_issue_is_concrete_and_not_recommended(self):
        result, _, _ = self.run_review(lambda data: data["items"][0]["review"].update(hard_issues=["新增未经确认的承诺"]))
        self.assertEqual(result.report.items[0].conclusion, "新增未经确认的承诺")
        self.assertEqual(result.report.items[0].verdict, "needs_attention")
        self.assertNotIn(0, result.report.recommended_candidate_indexes)

    def test_known_defect_is_not_hidden_behind_missing_information(self):
        result, _, _ = self.run_review(lambda data: data["items"][0]["review"].update(
            naturalness=3, critical_unknowns=["能否明天赴约未知"], issues=["语气不够自然"]))
        self.assertEqual(result.report.items[0].verdict, "needs_attention")
        self.assertEqual(result.report.items[0].uncertainties, ["能否明天赴约未知"])
        self.assertNotIn(0, result.report.recommended_candidate_indexes)

    def test_partial_group_never_gets_overall_recommendation(self):
        result, _, _ = self.run_review(lambda data: data["group_review"].update(reason=""))
        self.assertEqual(result.report.status, "partial")
        self.assertEqual(result.report.recommended_candidate_indexes, [])
        self.assertEqual(result.report.recommendation_reason, "")

    def test_recommendation_prioritizes_weakest_core_then_sum(self):
        def mutate(data):
            data["items"][0]["review"].update(naturalness=5, relevance=5, relation_fit=5, scenario_fit=5)
            data["items"][1]["review"].update(relevance=5)
        result, _, _ = self.run_review(mutate)
        self.assertEqual(result.report.recommended_candidate_indexes, [0])


class ReviewModeTests(TestCase):
    def test_routing_conservatively_keeps_risks_in_full_mode(self):
        args = [ChatAnalysis(risk_level="low"), ChatScenario(), ReplyControls(), ExtractedChat()]
        settings = Settings(_env_file=None, quality_quick_enabled=True)
        with patch("chat_reply_agent.services.quality_queue_service.get_settings", return_value=settings):
            self.assertEqual(select_review_mode(*args), "quick")
            for index, value in [(0, ChatAnalysis(risk_level="high")),
                                 (1, ChatScenario(type="conflict")), (1, ChatScenario(type="boundary")),
                                 (1, ChatScenario(type="apology")), (2, ReplyControls(intent="decline")),
                                 (2, ReplyControls(intent="cool_down")), (3, ExtractedChat(review_required=True)),
                                 (3, ExtractedChat(messages=[ChatMessage(text="谁说的", speaker="unknown")]))]:
                changed = list(args)
                changed[index] = value
                self.assertEqual(select_review_mode(*changed), "full")
            self.assertEqual(select_review_mode(*args, weather_evidence=[{}]), "full")
            settings.quality_quick_enabled = False
            self.assertEqual(select_review_mode(*args), "full")

    def test_quick_budget_restores_full_and_never_changes_generation(self):
        settings = Settings(_env_file=None, openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                            openai_model_name="qwen3.8-max")
        with patch("chat_reply_agent.services.llm_client.get_settings", return_value=settings), model_role("review"):
            self.assertEqual(_quality_options(), {})
            with quality_model_limits():
                self.assertEqual(_quality_options()["extra_body"]["thinking_budget"], 2048)
                with quality_model_limits("quick"):
                    self.assertEqual(_quality_options(), {"max_completion_tokens": 3072, "extra_body": {"thinking_budget": 512}})
                self.assertEqual(_quality_options()["extra_body"]["thinking_budget"], 2048)
            self.assertEqual(_quality_options(), {})


class QuickQueueTests(QueueHarness, TestCase):
    def setUp(self):
        self.start_queue(workers=1)

    def test_quick_original_uses_full_recheck_and_preserves_useful_summary(self):
        payload = {**review_input(), "review_mode": "quick"}
        texts = [reply.text for reply in payload["replies"]]
        initial = reviewed_payload(texts)
        initial["items"][0]["review"]["hard_issues"] = ["新增承诺"]
        revised = ["好，那先这样", *texts[1:]]
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=judge_responses(initial, reviewed_payload(revised))) as judge, patch(
                "chat_reply_agent.services.quality_repair_service.call_text_model", return_value='{"items":[{"candidate_index":0,"text":"好，那先这样"}]}'):
            job, _ = self.manager.submit("owner", payload, 90)
            state = self.terminal(job)
        self.assertEqual(state.quality.review_mode, "quick")
        self.assertEqual(state.revision_quality.review_mode, "full")
        self.assertIn("1 条建议调整", state.quality.summary)
        self.assertIn("精简评审", judge.call_args_list[0].args[1])
        self.assertIn("对照例子", judge.call_args_list[1].args[1])
        self.assertEqual(state.diagnostics["review_mode"], "quick")
        self.assertEqual(state.diagnostics["phase_review_mode"], "full")

    def test_quick_and_full_requests_do_not_deduplicate(self):
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=self.blocked_review):
            first, _ = self.manager.submit("owner", {**review_input(), "review_mode": "quick"}, 90)
            self.assertTrue(self.started.wait(1))
            second, _ = self.manager.submit("owner", {**review_input(), "review_mode": "full"}, 90)
            self.assertNotEqual(first, second)
            self.release.set()
            self.terminal(first)
            self.terminal(second)
