import json
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from fixtures_quality import judge_responses, reviewed_payload
from fixtures_v021 import GENERATION
from chat_reply_agent.schemas import ChatAnalysis, ChatScenario, ReplyCandidate, ReplyControls
from chat_reply_agent.services.generation_service import understand_and_generate
from chat_reply_agent.services.quality_repair_service import repair_candidates, repair_targets
from chat_reply_agent.services.quality_service import review_candidates
from chat_reply_agent.services.reply_policy import BASE_REPLY_POLICY, relationship_context, scenario_catalog
from chat_reply_agent.services.reply_service import (
    generate_reply_candidates, normalize_reply_candidates, regenerate_reply_candidate,
)


COMPLETE_REPLY = (
    "我确实有疑问，我这边显示的是被其他应用中断，但这也不能说明一定有人给你打电话。\n\n"
    "我想弄清楚发生了什么，不是一定要你交截图。\n\n"
    "你不想发可以不发，但你说‘那你怀疑吧’，这个问题就没法往下说了。"
)


class ReplyPolicyTests(TestCase):
    def setUp(self):
        self.rules = {"name": "恋人", "stage": "已确认关系", "boundaries": ["尊重隐私"],
                      "instructions": "LEGACY_ALWAYS_APOLOGIZE", "reply_rules": ["LEGACY_SHORT_ONLY"]}
        self.scenario = ChatScenario(type="conflict", reply_goal="LEGACY_ALWAYS_SOOTHE")
        self.replies = [ReplyCandidate(text=text, style="自然", reason="测试")
                        for text in (COMPLETE_REPLY, "我想先把这件事说清楚", "这件事我还想再聊聊")]

    def test_default_is_adaptive_without_breaking_saved_length_preferences(self):
        self.assertEqual(ReplyControls().length_preference, "auto")
        for length in ("short", "very_short", "normal"):
            self.assertEqual(ReplyControls(length_preference=length).length_preference, length)

    def test_metadata_does_not_include_legacy_strategy_bodies(self):
        self.assertEqual(set(relationship_context(self.rules)), {"name", "stage", "boundaries"})
        scenarios = {"conflict": {"label": "冲突", "description": "争执", "reply_goal": "LEGACY"}}
        self.assertEqual(scenario_catalog(scenarios), [{"type": "conflict", "label": "冲突", "description": "争执"}])

    def test_generation_preserves_complete_paragraphs_and_avoids_eager_rules(self):
        data = json.loads(GENERATION)
        data["replies"][0]["text"] = COMPLETE_REPLY
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=json.dumps(data)) as model:
            result = understand_and_generate("对方：那你怀疑吧", self.rules, "", "")
        self.assertEqual(result.replies[0].text, COMPLETE_REPLY)
        self.assertIn(BASE_REPLY_POLICY, model.call_args.args[0])
        self.assertNotIn("LEGACY_", model.call_args.args[1])
        self.assertNotIn('"natural_examples"', model.call_args.args[1])

    def test_legacy_and_single_generation_share_policy_and_keep_paragraphs(self):
        args = ("对方：那你怀疑吧", ChatAnalysis(), self.scenario, self.rules, "", "", ReplyControls())
        with patch("chat_reply_agent.services.reply_service.call_text_model", return_value=json.dumps(
                {"replies": [reply.model_dump() for reply in self.replies]})) as model:
            replies = generate_reply_candidates(*args)
        self.assertEqual(replies[0].text, COMPLETE_REPLY)
        self.assertNotIn("LEGACY_", model.call_args.args[1])
        self.assertIn(BASE_REPLY_POLICY, model.call_args.args[0])
        replacement = COMPLETE_REPLY + "\n\n我想听听你怎么看。"
        with patch("chat_reply_agent.services.reply_service.call_text_model", return_value=json.dumps(
                {"reply": {"text": replacement, "style": "说明", "reason": "测试"}})) as model:
            reply = regenerate_reply_candidate(*args, self.replies, 0)
        self.assertEqual(reply.text, replacement)
        self.assertNotIn("LEGACY_", model.call_args.args[1])

    def review(self, payload, *, controls=None):
        with patch("chat_reply_agent.services.quality_service.call_text_model",
                   side_effect=judge_responses(payload)) as model:
            result = review_candidates("对方：那你怀疑吧", self.replies, self.rules, "", "",
                controls or ReplyControls(), {i: ["完整检查"] for i in range(3)}, audit_only=True)
        self.assertNotIn("LEGACY_", model.call_args.args[1])
        self.assertIn(BASE_REPLY_POLICY, model.call_args.args[1])
        return result

    def test_length_punctuation_and_ordinary_words_do_not_cap_model_judgment(self):
        payload = reviewed_payload([reply.text for reply in self.replies])
        result = self.review(payload, controls=ReplyControls(length_preference="short"))
        item = result.report.items[0]
        self.assertEqual(item.naturalness, 4)
        self.assertIsNone(item.local_naturalness)
        self.assertEqual(item.verdict, "usable")
        self.assertTrue(item.advisories)
        self.assertFalse(item.issues)
        self.assertEqual(item.evaluated_text, COMPLETE_REPLY)
        self.assertNotIn(0, repair_targets(result.report, self.replies, 80))

    def test_firmness_is_not_automatic_pass_and_evidenced_defects_still_repair(self):
        payload = reviewed_payload([reply.text for reply in self.replies])
        review = payload["items"][0]["review"]
        review["relevance"] = 2
        review["issues"] = ["测试评审指出候选漏答本轮核心问题"]
        result = self.review(payload)
        self.assertEqual(result.report.items[0].verdict, "needs_attention")
        self.assertIn(0, repair_targets(result.report, self.replies, 80))

    def test_paragraphs_survive_normalization_and_independent_repair(self):
        normalized = normalize_reply_candidates(self.replies, self.rules, self.scenario)
        self.assertEqual(normalized[0].text, COMPLETE_REPLY)
        payload = {"chat_text": "对方：那你怀疑吧", "relation_rules": self.rules,
                   "memory_summary": "", "style_summary": "", "controls": ReplyControls(), "replies": self.replies}
        report = self.review(reviewed_payload([reply.text for reply in self.replies])).report
        changed = COMPLETE_REPLY + "\n\n截图的事先不说了。"
        before = deepcopy(self.replies)
        with patch("chat_reply_agent.services.quality_repair_service.call_text_model", return_value=json.dumps(
                {"items": [{"candidate_index": 0, "text": changed}]})) as model:
            revised = repair_candidates(payload, report, {0: ["补充回应"]})
        self.assertEqual(self.replies, before)
        self.assertEqual(revised[0].text, changed)
        self.assertIn(BASE_REPLY_POLICY, model.call_args.args[0])
        self.assertNotIn("LEGACY_", model.call_args.args[1])

    def test_long_unreviewed_text_has_no_invented_quality_score(self):
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=None):
            result = review_candidates("对话", self.replies, self.rules, "", "", ReplyControls(), {})
        self.assertTrue(all(item.local_naturalness is None and item.overall_score is None for item in result.report.items))
        self.assertFalse(result.report.passed)
