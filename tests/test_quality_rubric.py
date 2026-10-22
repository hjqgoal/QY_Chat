import json
from unittest import TestCase
from unittest.mock import patch

from fixtures_quality import DIMENSIONS, reviewed_payload
from chat_reply_agent.schemas import ChatAnalysis, ChatScenario, ReplyCandidate, ReplyControls, ReplyQualityItem
from chat_reply_agent.services.directed_review_service import review_targeted_replies
from chat_reply_agent.services.evaluation_service import load_evaluation_dataset
from chat_reply_agent.services.quality_service import review_and_refine_replies, review_candidates, similar_candidate_pairs
from chat_reply_agent.services.skill_loader import load_quality_skill


class QualityRubricTests(TestCase):
    def setUp(self):
        self.replies = [ReplyCandidate(text=text, style="生成标签不可作证据", reason="生成理由不可作证据")
                        for text in ("那你先试试", "能联机再叫我", "行，试完跟我说下")]
        self.kwargs = dict(chat_text="对方：我还能小试一下", replies=self.replies,
                           analysis=ChatAnalysis(tone="上轮推断不可作证据"),
                           scenario=ChatScenario(type="daily_chat", confidence=0.9),
                           relation_rules={"relation_type": "friend"}, memory_summary="",
                           style_summary="", controls=ReplyControls())

    def payload(self):
        return reviewed_payload([r.text for r in self.replies])

    def run_review(self, payload=None, **options):
        with patch("chat_reply_agent.services.quality_service.call_text_model",
                   return_value=json.dumps(payload or self.payload())) as model:
            result = review_and_refine_replies(**self.kwargs, **options)
        return result, model

    def test_online_review_has_evidence_and_no_numeric_badge_score(self):
        result, model = self.run_review()
        model.assert_called_once()
        self.assertTrue(result.report.passed)
        self.assertIsNone(result.report.average_score)
        for item, reply in zip(result.report.items, self.replies):
            self.assertEqual(item.verdict, "usable")
            self.assertIsNone(item.overall_score)
            self.assertEqual(item.evaluated_text, reply.text)
            self.assertEqual(set(item.evidence), set(DIMENSIONS))
            self.assertIsNone(item.style_fit)

    def test_skill_body_examples_and_primary_evidence_are_loaded(self):
        self.kwargs["memory_summary"] = "用户证实的旧事"
        self.kwargs["style_summary"] = "本人原句样本：行，我看看"
        result, model = self.run_review()
        prompt = model.call_args.args[1]
        for text in ("五档的含义", "对方收回邀请", "group_review", "用户证实的旧事", "本人原句样本"):
            self.assertIn(text, prompt)
        for text in ("生成标签不可作证据", "生成理由不可作证据", "上轮推断不可作证据"):
            self.assertNotIn(text, prompt)
        self.assertIn("改写后的文本", model.call_args.args[0])
        self.assertEqual(result.report.items[0].style_fit, 4)
        self.assertIn(load_quality_skill(), prompt)

    def test_unknown_core_dimension_does_not_get_a_default_grade(self):
        payload = self.payload()
        payload["items"][0]["review"]["scenario_fit"] = None
        payload["items"][0]["review"]["evidence"]["scenario_fit"] = "缺少最后一句的语气背景。"
        result, _ = self.run_review(payload, audit_only=True)
        self.assertEqual(result.report.items[0].verdict, "uncertain")
        self.assertIsNone(result.report.items[0].overall_score)
        self.assertIsNone(result.report.average_score)
        self.assertFalse(result.report.passed)

    def test_preference_only_context_is_not_personal_style_evidence(self):
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=json.dumps(self.payload())) as model:
            result = review_candidates("对方：我还能小试一下", self.replies, {}, "", "用户不喜欢：太长 2 次",
                ReplyControls(), {i: ["完整评估"] for i in range(3)}, style_evidence_available=False)
        self.assertTrue(all(item.style_fit is None for item in result.report.items))
        self.assertIn('"has_style_evidence": false', model.call_args.args[1])

    def test_hard_issue_or_uncertainty_cannot_hide_behind_perfect_scores(self):
        for key in ("hard_issues", "uncertainties"):
            payload = self.payload()
            review = payload["items"][0]["review"]
            review.update({dimension: 5 for dimension in DIMENSIONS})
            review[key] = ["新增时间承诺或承诺条件未确认。"]
            result, _ = self.run_review(payload, audit_only=True)
            self.assertFalse(result.report.passed)
            self.assertIsNone(result.report.items[0].overall_score)
            self.assertIsNone(result.report.average_score)

    def test_offline_only_scores_original_and_discloses_missing_style(self):
        result, model = self.run_review(audit_only=True)
        self.assertEqual(result.replies, self.replies)
        self.assertEqual(result.report.average_score, 80)
        self.assertEqual(result.report.items[0].assessment_source, "read_only")
        self.assertIn("四项", result.report.summary)
        self.assertIn("禁止改写", model.call_args.args[1])

    def test_weak_or_missing_group_review_cannot_pass_the_group(self):
        for group in ({}, {"distinct": True, "reason": "差异明显", "evaluated_texts": ["旧文本"]},
                      {"distinct": "true", "reason": "差异明显", "evaluated_texts": [r.text for r in self.replies]}):
            payload = self.payload()
            payload["group_review"] = group
            result, _ = self.run_review(payload)
            self.assertTrue(all(item.passed for item in result.report.items))
            self.assertFalse(result.report.passed)
            self.assertEqual(result.report.diversity_status, "not_reviewed")

    def test_empty_evidence_and_mixed_text_grades_fail_closed(self):
        for mutate in (lambda row: row["review"]["evidence"].update(relevance=" "),
                       lambda row: row["review"].update(evaluated_text="另一个答案")):
            payload = self.payload()
            mutate(payload["items"][1])
            result, _ = self.run_review(payload)
            self.assertEqual(result.report.status, "invalid_response")
            self.assertEqual(result.replies, self.replies)
            self.assertTrue(all(not item.passed for item in result.report.items))

    def test_batch_edits_colliding_with_each_other_are_rejected(self):
        payload = self.payload()
        payload["items"][0]["reply"]["text"] = "行，那先看看"
        payload["items"][1]["reply"]["text"] = "行，那先看看"
        result, _ = self.run_review(payload)
        self.assertEqual(result.replies, self.replies)
        self.assertEqual(result.report.items[0].status, "invalid_response")
        self.assertEqual(result.report.items[1].status, "invalid_response")

    def test_real_invitation_example_flags_reordered_candidates_in_one_call(self):
        _, _, cases = load_evaluation_dataset()
        case = next(case for case in cases if case.id == "friend_withdrawn_invitation")
        texts = ["行，那我先去了，要是缺人再说", "好啦，那我先去了，你们玩开心", "那你们玩得开心，我先去了"]
        replies = [ReplyCandidate(text=text, style="自然", reason="") for text in texts]
        self.assertIn((1, 2), similar_candidate_pairs(replies))
        payload = reviewed_payload(texts)
        payload["group_review"].update(distinct=False, reason="候选2和3都在祝玩得开心后收尾，只有语序变化。")
        indexes = {i for pair in similar_candidate_pairs(replies) for i in pair}
        payload["items"] = [row for row in payload["items"] if row["candidate_index"] in indexes]
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=json.dumps(payload)) as model:
            result = review_targeted_replies(case.expected_chat().to_text(), replies,
                ChatAnalysis(), ChatScenario(type="daily_chat", confidence=0.9), {}, "", "", ReplyControls())
        model.assert_called_once()
        self.assertEqual(result.report.diversity_status, "similar")
        self.assertFalse(result.report.passed)
        self.assertTrue(all(item.overall_score is None for item in result.report.items))
        self.assertIn("语序变化", result.report.group_issues[0])

    def test_legacy_reports_are_not_relabelled_as_new_assessments(self):
        item = ReplyQualityItem(candidate_index=0, status="reviewed", overall_score=100, passed=True)
        self.assertEqual(item.assessment_source, "legacy")
        self.assertEqual(item.verdict, "unverified")
