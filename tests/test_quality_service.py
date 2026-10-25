import json
from unittest import TestCase
from unittest.mock import patch

from chat_reply_agent.schemas import ChatAnalysis, ChatScenario, ReplyCandidate, ReplyControls
from chat_reply_agent.services.llm_client import ModelCallError
from chat_reply_agent.services.quality_service import (
    review_and_refine_replies, review_and_refine_single_reply,
)
from chat_reply_agent.services.reply_service import normalize_reply_candidates
from fixtures_quality import reviewed_payload


class QualityServiceTests(TestCase):
    def setUp(self):
        self.replies = [ReplyCandidate(text=text, style="自然", reason="") for text in (
            "那你先试试", "能联机再叫我", "行，试完跟我说下",
        )]
        self.kwargs = dict(
            chat_text="对方：我还能小试一下", analysis=ChatAnalysis(),
            scenario=ChatScenario(type="daily_chat"), relation_rules={},
            memory_summary="", style_summary="", controls=ReplyControls(),
        )

    def review(self, raw=None, error=None):
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=raw, side_effect=error):
            return review_and_refine_replies(replies=self.replies, **self.kwargs)

    def test_unavailable_and_invalid_reviews_preserve_text_without_scores(self):
        for raw, error, expected in (
            (None, None, "unavailable"),
            (None, ModelCallError("error"), "unavailable"),
            ("", None, "invalid_response"),
            ("not json", None, "invalid_response"),
            ('{"replies":[]}', None, "invalid_response"),
        ):
            with self.subTest(raw=raw, error=error):
                result = self.review(raw, error)
                self.assertEqual(result.replies, self.replies)
                self.assertEqual(result.report.status, expected)
                self.assertIsNone(result.report.average_score)
                self.assertFalse(result.report.passed)
                for item in result.report.items:
                    self.assertIsNone(item.overall_score)
                    self.assertIsNone(item.relevance)
                    self.assertIsNone(item.local_naturalness)

    def test_rewritten_text_cannot_inherit_the_original_assessment(self):
        payload = reviewed_payload(["先试试能不能联机", "能一起玩再说", "行，试好了叫我"],
                                   [item.text for item in self.replies])
        result = self.review(json.dumps(payload))
        self.assertEqual(result.report.status, "partial")
        self.assertFalse(result.report.passed)
        self.assertIsNone(result.report.average_score)
        self.assertEqual([item.text for item in result.replies], [item["reply"]["text"] for item in payload["items"]])
        self.assertTrue(all(item.revised for item in result.report.items))
        for item in result.report.items:
            self.assertEqual(item.status, "revised_pending")
            self.assertIsNone(item.overall_score)
            self.assertIsNone(item.evaluated_text)
            self.assertEqual(item.evidence, {})
        self.assertNotEqual(result.report.diversity_status, "distinct")

    def test_missing_invalid_or_duplicate_dimensions_do_not_get_default_scores(self):
        for bad_value in ("5", 5.0, True, 0, 6):
            payload = reviewed_payload([item.text for item in self.replies])
            payload["items"][0]["review"]["relevance"] = bad_value
            with self.subTest(value=bad_value):
                self.assertEqual(self.review(json.dumps(payload)).report.status, "invalid_response")
        for mutation in ("duplicate", "missing", "bad_reply", "bad_issues", "missing_evidence", "wrong_text"):
            payload = reviewed_payload([item.text for item in self.replies])
            if mutation == "duplicate":
                payload["items"][1]["candidate_index"] = 0
            elif mutation == "missing":
                del payload["items"][0]["review"]["style_fit"]
            elif mutation == "bad_reply":
                payload["items"][0]["reply"]["text"] = None
            elif mutation == "bad_issues":
                payload["items"][0]["review"]["issues"] = "invalid"
            elif mutation == "missing_evidence":
                del payload["items"][0]["review"]["evidence"]["relevance"]
            elif mutation == "wrong_text":
                payload["items"][0]["review"]["evaluated_text"] = "不是输入原文"
            with self.subTest(mutation=mutation):
                result = self.review(json.dumps(payload))
                self.assertIsNone(result.report.average_score)
                self.assertEqual(result.report.status, "invalid_response")
                self.assertEqual(result.replies, self.replies)

    def test_low_relevance_or_duplicate_text_cannot_pass_on_high_average(self):
        payload = reviewed_payload([item.text for item in self.replies])
        payload["items"][0]["review"]["relevance"] = 1
        self.assertFalse(self.review(json.dumps(payload)).report.passed)
        for reply in self.replies:
            reply.text = "先试试吧"
        payload = reviewed_payload([item.text for item in self.replies])
        result = self.review(json.dumps(payload))
        self.assertFalse(result.report.passed)
        self.assertEqual(result.report.diversity_status, "similar")
        self.assertIn("完全相同", result.report.group_issues[0])

    def test_fallbacks_are_not_counted_as_successful_generation(self):
        self.replies[0].source = "fallback"
        payload = reviewed_payload([item.text for item in self.replies])
        payload["items"] = payload["items"][1:]
        result = self.review(json.dumps(payload))
        self.assertIsNone(result.report.items[0].overall_score)
        self.assertIsNone(result.report.average_score)
        self.assertEqual(result.report.status, "partial")

    def test_length_and_style_words_no_longer_trigger_template_replacement(self):
        texts = [
            "我主要是想和你一起玩，国际服能不能联机还不确定，所以先别急着充值，等你试完能不能一起玩，再决定要不要买。",
            "如果你愿意试就试试，能联机再说",
            "我理解你的感受，我们可以慢慢沟通",
        ]
        result = normalize_reply_candidates(
            [{"text": text, "style": "自然", "reason": ""} for text in texts],
            {}, ChatScenario(type="daily_chat"),
        )
        self.assertEqual([item.text for item in result], texts)
        self.assertTrue(all(item.source == "model" for item in result))
        self.replies = result
        unreviewed = self.review()
        self.assertEqual(unreviewed.replies, result)
        self.assertEqual(unreviewed.report.items[2].issues, ["模型复核未完成，仅保留本地规则提醒。"])
        self.assertIsNone(unreviewed.report.items[2].local_naturalness)

    def test_malformed_candidate_is_explicit_fallback(self):
        result = normalize_reply_candidates([{"text": None}], {}, ChatScenario())
        self.assertEqual(len(result), 3)
        self.assertTrue(all(item.source == "fallback" and item.warnings for item in result))

    def test_single_review_failure_and_duplicate_revert_cannot_reuse_scores(self):
        payload = reviewed_payload([item.text for item in self.replies])
        payload["items"] = payload["items"][:1]
        payload["items"][0]["reply"]["text"] = self.replies[1].text
        for data in (None, json.dumps(payload)):
            with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=data):
                reply, quality = review_and_refine_single_reply(
                    reply=self.replies[0], candidate_index=0,
                    existing_replies=self.replies, **self.kwargs,
                )
            self.assertEqual(reply, self.replies[0])
            self.assertIsNone(quality.overall_score)
            self.assertFalse(quality.passed)
