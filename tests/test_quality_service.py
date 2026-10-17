import json
from unittest import TestCase
from unittest.mock import patch

from chat_reply_agent.schemas import ChatAnalysis, ChatScenario, ReplyCandidate, ReplyControls
from chat_reply_agent.services.llm_client import ModelCallError
from chat_reply_agent.services.quality_service import (
    review_and_refine_replies, review_and_refine_single_reply,
)
from chat_reply_agent.services.reply_service import normalize_reply_candidates


def reviewed_payload(texts):
    return {
        "replies": [{"text": text, "style": "自然", "reason": "承接当前话题"} for text in texts],
        "reviews": [{
            "candidate_index": index, "naturalness": 5, "relevance": 5,
            "relation_fit": 5, "scenario_fit": 5, "style_fit": 5, "issues": [],
        } for index in range(3)],
    }


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
                    self.assertIsNotNone(item.local_naturalness)

    def test_complete_review_scores_exact_final_text(self):
        payload = reviewed_payload(["先试试能不能联机", "能一起玩再说", "行，试好了叫我"])
        result = self.review(json.dumps(payload))
        self.assertEqual(result.report.status, "reviewed")
        self.assertTrue(result.report.passed)
        self.assertEqual(result.report.average_score, 100)
        self.assertEqual([item.text for item in result.replies], [item["text"] for item in payload["replies"]])
        self.assertTrue(all(item.revised for item in result.report.items))

    def test_missing_invalid_or_duplicate_dimensions_do_not_get_default_scores(self):
        for bad_value in (None, "5", 5.0, True, 0, 6):
            payload = reviewed_payload([item.text for item in self.replies])
            payload["reviews"][0]["relevance"] = bad_value
            with self.subTest(value=bad_value):
                self.assertIsNone(self.review(json.dumps(payload)).report.average_score)
        for mutation in ("duplicate", "missing", "bad_reply", "bad_issues"):
            payload = reviewed_payload([item.text for item in self.replies])
            if mutation == "duplicate":
                payload["reviews"][1]["candidate_index"] = 0
            elif mutation == "missing":
                del payload["reviews"][0]["style_fit"]
            elif mutation == "bad_reply":
                payload["replies"][0]["text"] = None
            else:
                payload["reviews"][0]["issues"] = "invalid"
            with self.subTest(mutation=mutation):
                result = self.review(json.dumps(payload))
                self.assertIsNone(result.report.average_score)
                self.assertEqual(result.replies, self.replies)

    def test_low_relevance_or_duplicate_text_cannot_pass_on_high_average(self):
        payload = reviewed_payload([item.text for item in self.replies])
        payload["reviews"][0]["relevance"] = 1
        self.assertFalse(self.review(json.dumps(payload)).report.passed)
        payload = reviewed_payload(["先试试吧"] * 3)
        result = self.review(json.dumps(payload))
        self.assertFalse(result.report.passed)
        self.assertIn("候选重复", result.report.items[0].issues[-1])

    def test_fallbacks_are_not_counted_as_successful_generation(self):
        self.replies[0].source = "fallback"
        result = self.review(json.dumps(reviewed_payload([item.text for item in self.replies])))
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
        self.assertTrue(unreviewed.report.items[2].issues)

    def test_malformed_candidate_is_explicit_fallback(self):
        result = normalize_reply_candidates([{"text": None}], {}, ChatScenario())
        self.assertEqual(len(result), 3)
        self.assertTrue(all(item.source == "fallback" and item.warnings for item in result))

    def test_single_review_failure_and_duplicate_revert_cannot_reuse_scores(self):
        for data in (None, json.dumps({
            "reply": {"text": self.replies[1].text},
            "review": reviewed_payload(["a", "b", "c"])["reviews"][0],
        })):
            with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=data):
                reply, quality = review_and_refine_single_reply(
                    reply=self.replies[0], candidate_index=0,
                    existing_replies=self.replies, **self.kwargs,
                )
            self.assertEqual(reply, self.replies[0])
            self.assertIsNone(quality.overall_score)
            self.assertFalse(quality.passed)
