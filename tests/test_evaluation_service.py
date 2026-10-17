from pathlib import Path
from unittest import TestCase

from chat_reply_agent.schemas import (
    ChatMessage,
    ChatScenario,
    EvalCase,
    EvalExpectation,
    ExtractedChat,
    ReplyCandidate,
    ReplyQualityReport,
)
from chat_reply_agent.services.evaluation_service import (
    DEFAULT_DATASET_PATH,
    load_evaluation_dataset,
    score_evaluation_output,
)


class EvaluationDatasetTests(TestCase):
    def test_default_dataset_is_valid_and_has_unique_cases(self) -> None:
        name, version, cases = load_evaluation_dataset(DEFAULT_DATASET_PATH)

        self.assertTrue(name)
        self.assertTrue(version)
        self.assertGreaterEqual(len(cases), 10)
        self.assertEqual(len(cases), len({case.id for case in cases}))
        self.assertTrue(all(case.messages for case in cases))
        self.assertTrue(all(case.expected.reference_replies for case in cases))

    def test_missing_dataset_raises_file_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_evaluation_dataset(Path("/tmp/not-a-real-chat-eval.yaml"))


class EvaluationScoringTests(TestCase):
    def test_missing_quality_score_cannot_turn_into_a_passing_evaluation(self) -> None:
        case = _build_case(image_path=None)
        result = score_evaluation_output(
            case=case, actual_chat=case.expected_chat(),
            scenario=ChatScenario(type="daily_chat"), replies=_good_replies(),
            quality=ReplyQualityReport(status="unavailable"),
        )
        self.assertIsNone(result.score)
        self.assertIsNone(result.metrics.reply_quality)
        self.assertFalse(result.passed)

    def test_natural_diverse_replies_pass_text_case(self) -> None:
        case = _build_case(image_path=None)
        result = score_evaluation_output(
            case=case,
            actual_chat=case.expected_chat(),
            scenario=ChatScenario(type="daily_chat", label="日常聊天"),
            replies=_good_replies(),
            quality=ReplyQualityReport(passed=True, average_score=88),
        )

        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.score, 75)
        self.assertEqual(result.metrics.scenario_accuracy, 100)
        self.assertIsNone(result.metrics.recognition_accuracy)

    def test_same_text_with_wrong_speaker_is_detected(self) -> None:
        case = _build_case(image_path="duplicate-speaker.png")
        actual = ExtractedChat(
            messages=[
                ChatMessage(speaker="other", side="right", text="其实差不多"),
                ChatMessage(speaker="other", side="left", text="其实差不多"),
            ]
        )
        result = score_evaluation_output(
            case=case,
            actual_chat=actual,
            scenario=ChatScenario(type="daily_chat", label="日常聊天"),
            replies=_good_replies(),
            quality=ReplyQualityReport(passed=True, average_score=88),
        )

        self.assertEqual(result.metrics.text_accuracy, 100)
        self.assertEqual(result.metrics.speaker_accuracy, 50)
        self.assertEqual(result.metrics.recognition_accuracy, 85)
        self.assertIn("截图说话人识别准确率不足 90%", result.issues)

    def test_forbidden_ai_phrase_fails_constraint(self) -> None:
        case = _build_case(image_path=None)
        replies = _good_replies()
        replies[0] = ReplyCandidate(
            text="我理解你的感受，我们可以慢慢沟通",
            style="温柔",
            reason="测试禁用词",
        )
        result = score_evaluation_output(
            case=case,
            actual_chat=case.expected_chat(),
            scenario=ChatScenario(type="daily_chat", label="日常聊天"),
            replies=replies,
            quality=ReplyQualityReport(passed=False, average_score=60),
        )

        self.assertFalse(result.passed)
        self.assertLess(result.metrics.constraint_compliance, 100)
        self.assertTrue(any("禁用表达" in issue for issue in result.issues))


def _build_case(image_path: str | None) -> EvalCase:
    return EvalCase(
        id="duplicate-speaker",
        name="重复文字说话人测试",
        relation_type="friend",
        image_path=image_path,
        messages=[
            ChatMessage(speaker="me", side="right", text="其实差不多"),
            ChatMessage(speaker="other", side="left", text="其实差不多"),
        ],
        expected=EvalExpectation(
            scenario_types=["daily_chat"],
            reference_replies=["那就先用以前的"],
        ),
    )


def _good_replies() -> list[ReplyCandidate]:
    return [
        ReplyCandidate(text="那就先用以前的", style="直接", reason="顺着当前话题"),
        ReplyCandidate(text="行，能用就不折腾了", style="轻松", reason="自然收束"),
        ReplyCandidate(text="那还挺省事的", style="随和", reason="轻松回应"),
    ]
