from unittest import TestCase
from unittest.mock import patch

from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from chat_reply_agent.database import Base
from chat_reply_agent.main import regenerate_single_reply
from chat_reply_agent.models import (
    ContactMemory,
    ConversationMemory,
    MemoryEpisode,
    ReplyFeedback,
    StyleExample,
    UserStyleProfile,
)
from chat_reply_agent.schemas import (
    ChatAnalysis,
    ChatMessage,
    ChatScenario,
    ExtractedChat,
    RegenerateReplyRequest,
    ReplyCandidate,
    ReplyControls,
    ReplyQualityItem,
)
from chat_reply_agent.services.reply_service import (
    describe_reply_controls,
    get_intent_conflict_issue,
    normalize_reply_candidates,
    regenerate_reply_candidate,
)
from chat_reply_agent.services.skill_loader import load_relation_skill


class ReplyControlsTests(TestCase):
    def setUp(self) -> None:
        self.analysis = ChatAnalysis(
            tone="轻松",
            emotion="平静",
            attitude="开放",
            scene="日常聊天",
        )
        self.scenario = ChatScenario(
            type="daily_chat",
            label="日常聊天",
            confidence=1,
        )
        self.rules = load_relation_skill("friend")

    def test_reply_controls_validate_supported_values(self) -> None:
        with self.assertRaises(ValidationError):
            ReplyControls(intent="promise_everything")
        with self.assertRaises(ValidationError):
            ReplyControls(intimacy_level=4)

    def test_control_description_contains_user_choices(self) -> None:
        controls = ReplyControls(
            intent="decline",
            custom_intent="这次不去，但别显得生气",
            length_preference="very_short",
            intimacy_level=1,
            initiative_level=1,
        )

        description = describe_reply_controls(controls)

        self.assertIn("这次不去，但别显得生气", description)
        self.assertIn("2-12", description)
        self.assertIn("克制一点", description)
        self.assertIn("只回应眼前内容", description)

    def test_decline_control_replaces_candidates_that_agree(self) -> None:
        controls = ReplyControls(intent="decline")
        replies = normalize_reply_candidates(
            [
                {"text": "好啊", "style": "直接", "reason": ""},
                {"text": "可以呀", "style": "轻松", "reason": ""},
                {"text": "那就这么定", "style": "温和", "reason": ""},
            ],
            self.rules,
            self.scenario,
            controls,
        )

        self.assertEqual(len({item.text for item in replies}), 3)
        self.assertTrue(
            all(not get_intent_conflict_issue(item.text, controls) for item in replies)
        )

    def test_custom_intent_can_infer_a_decline_stance(self) -> None:
        controls = ReplyControls(
            intent="auto",
            custom_intent="帮我婉拒这次邀请",
        )

        replies = normalize_reply_candidates(
            [{"text": "好啊", "style": "直接", "reason": ""}],
            self.rules,
            self.scenario,
            controls,
        )

        self.assertNotEqual(replies[0].text, "好啊")
        self.assertFalse(get_intent_conflict_issue(replies[0].text, controls))

    @patch("chat_reply_agent.services.reply_service.call_text_model")
    def test_single_regeneration_avoids_all_existing_candidates(
        self,
        mock_model,
    ) -> None:
        mock_model.return_value = (
            '{"reply":{"text":"这次可能不行","style":"直接","reason":"换个说法"}}'
        )
        controls = ReplyControls(intent="decline")
        existing = [
            ReplyCandidate(text="这次可能不行", style="直接", reason=""),
            ReplyCandidate(text="我先不过去了", style="轻松", reason=""),
            ReplyCandidate(text="今天不太方便", style="温和", reason=""),
        ]

        reply = regenerate_reply_candidate(
            chat_text="对方：今晚一起吃饭吗",
            analysis=self.analysis,
            scenario=self.scenario,
            relation_rules=self.rules,
            memory_summary="",
            style_summary="",
            controls=controls,
            existing_replies=existing,
            candidate_index=0,
        )

        self.assertNotIn(reply.text, {item.text for item in existing})
        self.assertFalse(get_intent_conflict_issue(reply.text, controls))


class SingleRegenerationPersistenceTests(TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    @patch("chat_reply_agent.main.enqueue_quality")
    @patch("chat_reply_agent.main.regenerate_reply_candidate")
    def test_single_regeneration_does_not_write_memory_or_feedback(
        self,
        mock_regenerate,
        mock_review,
    ) -> None:
        generated = ReplyCandidate(
            text="这次先不去啦",
            style="自然",
            reason="保持拒绝立场",
        )
        quality = ReplyQualityItem(
            candidate_index=1,
            overall_score=88,
            naturalness=5,
            relevance=4,
            relation_fit=4,
            scenario_fit=5,
            style_fit=4,
            passed=True,
        )
        mock_regenerate.return_value = generated
        payload = RegenerateReplyRequest(
            relation_type="friend",
            user_id="v015-user",
            contact_id="xiaolin",
            contact_name="小林",
            background_context="我们之前已经推迟过一次见面。",
            extracted_chat=ExtractedChat(
                messages=[
                    ChatMessage(speaker="other", side="left", text="今晚出来吗"),
                ],
                topic="今晚邀约",
                last_message_from_other="今晚出来吗",
                confidence=1,
            ),
            analysis=ChatAnalysis(scene="邀约"),
            scenario=ChatScenario(type="invitation", label="邀约", confidence=1),
            existing_replies=[
                ReplyCandidate(text="今晚可能不行", style="直接", reason=""),
                ReplyCandidate(text="我这次就不去啦", style="轻松", reason=""),
                ReplyCandidate(text="今天有点不方便", style="温和", reason=""),
            ],
            candidate_index=1,
            controls=ReplyControls(intent="decline"),
        )
        models = (
            ConversationMemory,
            ContactMemory,
            MemoryEpisode,
            UserStyleProfile,
            StyleExample,
            ReplyFeedback,
        )
        before = {model: self.db.query(model).count() for model in models}
        from chat_reply_agent.schemas import QualityJobResponse, ReplyQualityReport
        updated = list(payload.existing_replies)
        updated[1] = generated
        mock_review.return_value = QualityJobResponse(job_id="job", status="pending", quality=ReplyQualityReport(
            items=[ReplyQualityItem(candidate_index=0), quality, ReplyQualityItem(candidate_index=2)]))

        response = regenerate_single_reply(payload, self.db)

        after = {model: self.db.query(model).count() for model in models}
        self.assertEqual(response.candidate_index, 1)
        self.assertEqual(response.reply.text, "这次先不去啦")
        self.assertIn("reply.regenerate", [item["stage"] for item in response.diagnostics["stages"]])
        self.assertIn("quality.enqueue", [item["stage"] for item in response.diagnostics["stages"]])
        self.assertEqual(response.quality_job_id, "job")
        self.assertEqual(before, after)
        mock_regenerate.assert_called_once()
        mock_review.assert_called_once()
        self.assertIn(
            "我们之前已经推迟过一次见面。",
            mock_regenerate.call_args.kwargs["chat_text"],
        )
