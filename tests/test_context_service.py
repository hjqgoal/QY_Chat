from contextlib import ExitStack
from unittest import TestCase
from unittest.mock import patch
from fixtures_v021 import GENERATION, MEMORY
from chat_reply_agent.services.memory_queue_service import process_next_memory_job

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from chat_reply_agent.database import Base
from chat_reply_agent.main import _build_reply_response
from chat_reply_agent.models import ConversationMemory, MemoryEpisode, UserStyleProfile
from chat_reply_agent.schemas import ChatMessage, ExtractedChat
from chat_reply_agent.services.context_service import (
    MAX_BACKGROUND_CONTEXT_LENGTH,
    build_context_retrieval_query,
    build_model_chat_context,
    normalize_background_context,
)


class ContextFormattingTests(TestCase):
    def test_normalizes_background_and_keeps_paragraph_breaks(self) -> None:
        value = "  昨天聊过见面。\r\n\r\n\r\n  今天对方主动联系。  "

        self.assertEqual(
            normalize_background_context(value),
            "昨天聊过见面。\n\n今天对方主动联系。",
        )

    def test_rejects_background_over_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "最多 2000 个字符"):
            normalize_background_context("背" * (MAX_BACKGROUND_CONTEXT_LENGTH + 1))

    def test_model_context_distinguishes_summary_from_recent_messages(self) -> None:
        chat = ExtractedChat(
            messages=[
                ChatMessage(speaker="other", side="left", text="那今晚见吗"),
            ],
            topic="确认见面",
        )

        context = build_model_chat_context(chat, "之前因为时间没谈拢。")

        self.assertIn("用户补充背景（概括，非逐字聊天）", context)
        self.assertIn("近期聊天（截图识别并由用户确认）", context)
        self.assertIn("冲突", context)
        self.assertLess(context.index("之前因为时间没谈拢"), context.index("那今晚见吗"))
        self.assertEqual(
            build_context_retrieval_query(chat, "之前因为时间没谈拢。"),
            "之前因为时间没谈拢。\n那今晚见吗",
        )

    def test_requires_background_or_chat_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "上传聊天截图或填写"):
            build_model_chat_context(
                ExtractedChat(messages=[], source_image_count=0),
                "",
            )


class ContextPipelineTests(TestCase):
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

    def test_background_reaches_model_steps_but_not_style_learning(self) -> None:
        background = "我们前几天因为见面时间闹得不太愉快。"
        chat = ExtractedChat(
            messages=[
                ChatMessage(speaker="other", side="left", text="晚上还见吗"),
                ChatMessage(speaker="me", side="right", text="我看看呀"),
            ],
            topic="再次确认见面",
        )
        targets = (
            "chat_reply_agent.services.generation_service.call_text_model",
        )
        calls = []
        with ExitStack() as stack:
            for target in targets:
                calls.append(stack.enter_context(patch(target, return_value=GENERATION)))
            response = _build_reply_response(
                db=self.db,
                user_id="context-user",
                relation_type="friend",
                contact_id="xiaolin",
                contact_name="小林",
                extracted_chat=chat,
                background_context=background,
                save_to_memory=False,
            )

        for model_call in calls:
            self.assertIn(background, model_call.call_args.args[1])
        self.assertEqual(response.background_context, background)
        self.assertNotIn(background, response.screenshot_text)
        self.assertEqual(response.style_evidence.current_examples, ["我看看呀"])

    def test_background_only_is_persisted_without_creating_style_profile(self) -> None:
        background = "我们认识半年，昨天对方说最近工作很累，我还没回复。"
        chat = ExtractedChat(messages=[], source_image_count=0)
        targets = (
            "chat_reply_agent.services.analysis_service.call_text_model",
            "chat_reply_agent.services.scenario_service.call_text_model",
            "chat_reply_agent.services.reply_service.call_text_model",
            "chat_reply_agent.services.quality_service.call_text_model",
            "chat_reply_agent.services.contact_memory_service.call_text_model",
        )
        with ExitStack() as stack:
            for target in targets:
                stack.enter_context(patch(target, return_value=None))
            stack.enter_context(patch("chat_reply_agent.services.generation_service.call_text_model", return_value=GENERATION))
            response = _build_reply_response(
                db=self.db,
                user_id="context-only-user",
                relation_type="friend",
                contact_id="xiaolin",
                contact_name="小林",
                extracted_chat=chat,
                background_context=background,
            )
            self.assertFalse(response.memory_persisted)
            self.assertEqual(response.memory_status, "pending")
            self.assertEqual(self.db.query(ConversationMemory).count(), 0)
            with patch("chat_reply_agent.services.contact_memory_service.call_text_model", return_value=MEMORY):
                process_next_memory_job(sessionmaker(bind=self.engine))

        memory = self.db.query(ConversationMemory).one()
        episode = self.db.query(MemoryEpisode).one()
        self.db.expire_all()
        self.assertIn(background, memory.last_chat_text)
        self.assertIn(background, episode.chat_text)
        self.assertEqual(response.style_evidence.current_user_message_count, 0)
        self.assertEqual(self.db.query(UserStyleProfile).count(), 0)
