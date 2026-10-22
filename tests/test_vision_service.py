import json
from unittest import TestCase
from unittest.mock import patch

from chat_reply_agent.main import _normalize_confirmed_chat
from chat_reply_agent.schemas import ChatMessage, ExtractedChat
from chat_reply_agent.services.llm_client import ModelCallError
from chat_reply_agent.services.vision_service import (
    _parse_extracted_chat,
    extract_chat_from_image,
    merge_extracted_chats,
)


class VisionNormalizationTests(TestCase):
    def test_empty_model_response_is_not_mistaken_for_missing_configuration(self) -> None:
        with patch("chat_reply_agent.services.vision_service.call_vision_model", return_value=""), self.assertRaises(ModelCallError):
            extract_chat_from_image("synthetic.png")

    def test_malformed_or_empty_output_is_not_presented_as_a_chat_message(self) -> None:
        for raw in ('```json\n{"messages": [\n```', '{"messages": []}', '{"messages": [{}]}', '{"messages": [{"text": null}]}'):
            with self.subTest(raw=raw), self.assertRaises(ModelCallError) as error:
                _parse_extracted_chat(raw)
            self.assertEqual(error.exception.kind, "invalid_structure")

    def test_missing_or_invalid_confidence_requires_review(self) -> None:
        for confidence in (None, "", "NaN", "Infinity", 98, -1):
            payload = {"messages": [{"side": "right", "text": "今晚来不了", "text_confidence": confidence,
                                     "position_confidence": confidence}], "confidence": 0.99}
            with self.subTest(confidence=confidence):
                extracted = _parse_extracted_chat(json.dumps(payload))
                self.assertTrue(extracted.review_required)
                self.assertEqual(extracted.messages[0].speaker, "me")
                self.assertLessEqual(extracted.messages[0].text_confidence, 0.5)
                self.assertLessEqual(extracted.messages[0].speaker_confidence, 0.5)

    def test_geometry_overrides_model_speaker_and_preserves_duplicates(self) -> None:
        payload = {
            "messages": [
                {
                    "speaker": "other",
                    "side": "left",
                    "bubble_center_x": 0.78,
                    "text": "其实差不多",
                    "text_confidence": 0.98,
                    "position_confidence": 0.96,
                },
                {
                    "speaker": "me",
                    "side": "right",
                    "bubble_center_x": 0.22,
                    "text": "其实差不多",
                    "text_confidence": 0.97,
                    "position_confidence": 0.95,
                },
            ],
            "topic": "比较差异",
            "confidence": 0.97,
            "last_message_from_other": "模型给错的最后一句",
        }

        extracted = _parse_extracted_chat(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(len(extracted.messages), 2)
        self.assertEqual(extracted.messages[0].speaker, "me")
        self.assertEqual(extracted.messages[0].side, "right")
        self.assertEqual(extracted.messages[1].speaker, "other")
        self.assertEqual(extracted.messages[1].side, "left")
        self.assertEqual(extracted.last_message_from_other, "其实差不多")
        self.assertTrue(extracted.review_required)
        self.assertIn("左右位置", "".join(extracted.messages[0].warnings))

    def test_low_confidence_message_is_marked_for_review(self) -> None:
        payload = {
            "messages": [
                {
                    "side": "left",
                    "bubble_center_x": 0.2,
                    "text": "这句有点看不清",
                    "text_confidence": 0.55,
                    "position_confidence": 0.92,
                },
                {
                    "side": "right",
                    "bubble_center_x": 0.8,
                    "text": "嗯",
                    "text_confidence": 0.96,
                    "position_confidence": 0.96,
                },
            ],
            "confidence": 0.8,
        }

        extracted = _parse_extracted_chat(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(extracted.review_message_indexes, [0])
        self.assertTrue(extracted.messages[0].needs_review)
        self.assertFalse(extracted.messages[1].needs_review)
        self.assertIn("文字识别置信度较低", "".join(extracted.messages[0].warnings))

    def test_side_is_used_when_center_position_is_missing(self) -> None:
        payload = {
            "messages": [
                {
                    "side": "左侧",
                    "text": "晚上吃什么",
                    "text_confidence": 0.95,
                    "position_confidence": 0.9,
                }
            ]
        }

        extracted = _parse_extracted_chat(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(extracted.messages[0].speaker, "other")
        self.assertEqual(extracted.messages[0].speaker_source, "side")
        self.assertFalse(extracted.review_required)


class ConfirmedChatTests(TestCase):
    def test_unknown_speaker_does_not_become_high_confidence_when_confirming_text(self) -> None:
        confirmed = _normalize_confirmed_chat(ExtractedChat(messages=[ChatMessage(speaker="unknown", text="今晚来不了")]), "已核对文字。")
        self.assertEqual(confirmed.messages[0].speaker_confidence, 0.0)
        self.assertEqual(confirmed.messages[0].speaker_source, "unknown")
        self.assertEqual(confirmed.review_message_indexes, [0])
        self.assertTrue(confirmed.review_required)
        self.assertEqual(confirmed.last_message_from_other, "")

    def test_confirmation_clears_review_flags_and_rebuilds_last_other(self) -> None:
        extracted = ExtractedChat(
            messages=[
                ChatMessage(
                    speaker="other",
                    side="right",
                    text="  你到了吗  ",
                    text_confidence=0.5,
                    speaker_confidence=0.4,
                    needs_review=True,
                    warnings=["需要复核"],
                ),
                ChatMessage(speaker="me", side="left", text="快了"),
            ],
            review_required=True,
            review_message_indexes=[0],
            source_image_count=3,
            merged_overlap_count=2,
        )

        confirmed = _normalize_confirmed_chat(extracted, "用户已确认。")

        self.assertEqual(confirmed.messages[0].side, "left")
        self.assertEqual(confirmed.messages[1].side, "right")
        self.assertEqual(confirmed.messages[0].text, "你到了吗")
        self.assertEqual(confirmed.last_message_from_other, "你到了吗")
        self.assertFalse(confirmed.review_required)
        self.assertEqual(confirmed.review_message_indexes, [])
        self.assertTrue(all(not item.needs_review for item in confirmed.messages))
        self.assertTrue(all(item.speaker_source == "user_confirmed" for item in confirmed.messages))
        self.assertEqual(confirmed.source_image_count, 3)
        self.assertEqual(confirmed.merged_overlap_count, 2)


class MultiScreenshotMergeTests(TestCase):
    def test_removes_only_suffix_prefix_overlap(self) -> None:
        first = ExtractedChat(
            messages=[
                ChatMessage(speaker="me", side="right", text="刚下班"),
                ChatMessage(speaker="other", side="left", text="吃饭了吗"),
                ChatMessage(speaker="me", side="right", text="还没呢"),
            ],
            topic="晚饭",
        )
        second = ExtractedChat(
            messages=[
                ChatMessage(speaker="other", side="left", text="吃饭了吗"),
                ChatMessage(speaker="me", side="right", text="还没呢"),
                ChatMessage(speaker="other", side="left", text="那一起吃？"),
            ],
            topic="约饭",
        )

        merged = merge_extracted_chats([first, second])

        self.assertEqual(
            [message.text for message in merged.messages],
            ["刚下班", "吃饭了吗", "还没呢", "那一起吃？"],
        )
        self.assertEqual(merged.source_image_count, 2)
        self.assertEqual(merged.merged_overlap_count, 2)
        self.assertEqual(merged.last_message_from_other, "那一起吃？")
        self.assertEqual(merged.topic, "")
        self.assertEqual(merged.topic_source, "pending")

    def test_preserves_real_duplicate_messages_inside_one_screenshot(self) -> None:
        chat = ExtractedChat(
            messages=[
                ChatMessage(speaker="me", side="right", text="其实差不多"),
                ChatMessage(speaker="other", side="left", text="其实差不多"),
            ]
        )

        merged = merge_extracted_chats([chat])

        self.assertEqual(len(merged.messages), 2)
        self.assertEqual(merged.merged_overlap_count, 0)

    def test_preserves_single_short_duplicate_at_screenshot_boundary(self) -> None:
        first = ExtractedChat(
            messages=[ChatMessage(speaker="other", side="left", text="好")]
        )
        second = ExtractedChat(
            messages=[ChatMessage(speaker="other", side="left", text="好")]
        )

        merged = merge_extracted_chats([first, second])

        self.assertEqual([message.text for message in merged.messages], ["好", "好"])
        self.assertEqual(merged.merged_overlap_count, 0)

    def test_removes_single_long_boundary_overlap(self) -> None:
        text = "那我们明天晚上见面吧"
        first = ExtractedChat(
            messages=[ChatMessage(speaker="other", side="left", text=text)]
        )
        second = ExtractedChat(
            messages=[
                ChatMessage(speaker="other", side="left", text=text),
                ChatMessage(speaker="me", side="right", text="行呀"),
            ]
        )

        merged = merge_extracted_chats([first, second])

        self.assertEqual([message.text for message in merged.messages], [text, "行呀"])
        self.assertEqual(merged.merged_overlap_count, 1)

    def test_rebuilds_review_indexes_after_overlap_is_removed(self) -> None:
        first = ExtractedChat(
            messages=[
                ChatMessage(speaker="me", side="right", text="第一条"),
                ChatMessage(
                    speaker="other",
                    side="left",
                    text="边界消息一",
                    needs_review=True,
                ),
                ChatMessage(speaker="me", side="right", text="边界消息二"),
            ]
        )
        second = ExtractedChat(
            messages=[
                ChatMessage(
                    speaker="other",
                    side="left",
                    text="边界消息一",
                    needs_review=True,
                ),
                ChatMessage(speaker="me", side="right", text="边界消息二"),
                ChatMessage(
                    speaker="other",
                    side="left",
                    text="最后一条",
                    needs_review=True,
                ),
            ]
        )

        merged = merge_extracted_chats([first, second])

        self.assertEqual(merged.review_message_indexes, [1, 3])
        self.assertTrue(merged.review_required)
        self.assertEqual(merged.last_message_from_other, "最后一条")
