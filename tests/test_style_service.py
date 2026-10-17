from unittest import TestCase

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from chat_reply_agent.database import Base
from chat_reply_agent.models import UserStyleProfile
from chat_reply_agent.schemas import ChatMessage, ExtractedChat
from chat_reply_agent.services.style_service import (
    build_current_user_style,
    get_saved_style_snapshot,
    get_style_summary,
    learn_user_style,
)


class StyleExtractionTests(TestCase):
    def test_extracts_only_messages_spoken_by_user(self) -> None:
        chat = ExtractedChat(
            messages=[
                ChatMessage(speaker="other", side="left", text="哈哈哈哈呀"),
                ChatMessage(speaker="me", side="right", text="行吧"),
                ChatMessage(speaker="me", side="right", text="晚点到"),
            ]
        )

        snapshot = build_current_user_style(chat)

        self.assertEqual(snapshot.message_count, 2)
        self.assertEqual(snapshot.phrases, ("行吧", "晚点到"))
        self.assertIn("吧", snapshot.summary)
        self.assertNotIn("笑声", snapshot.summary)
        self.assertNotIn("哈哈哈哈呀", snapshot.prompt_context("当前聊天"))

    def test_empty_user_side_has_no_style_evidence(self) -> None:
        chat = ExtractedChat(
            messages=[
                ChatMessage(speaker="other", side="left", text="你到了吗"),
                ChatMessage(speaker="unknown", side="unknown", text="看不清"),
            ]
        )

        snapshot = build_current_user_style(chat)

        self.assertEqual(snapshot.message_count, 0)
        self.assertEqual(snapshot.summary, "")
        self.assertEqual(snapshot.phrases, ())
        self.assertEqual(snapshot.prompt_context("当前聊天"), "")


class StylePersistenceTests(TestCase):
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

    def test_accumulates_and_deduplicates_user_phrases(self) -> None:
        first = ExtractedChat(
            messages=[
                ChatMessage(speaker="me", side="right", text="行呀"),
                ChatMessage(speaker="other", side="left", text="那就好"),
            ]
        )
        second = ExtractedChat(
            messages=[
                ChatMessage(speaker="me", side="right", text="晚点到"),
                ChatMessage(speaker="me", side="right", text="行呀"),
            ]
        )

        learn_user_style(self.db, "owner", "friend", first)
        profile = learn_user_style(self.db, "owner", "friend", second)
        snapshot = get_saved_style_snapshot(self.db, "owner", "friend")

        self.assertIsNotNone(profile)
        self.assertEqual(set(snapshot.phrases), {"行呀", "晚点到"})
        self.assertEqual(len(snapshot.phrases), 2)
        self.assertNotIn("那就好", snapshot.phrases)
        self.assertIn("用户本人原句样本", get_style_summary(self.db, "owner", "friend"))

    def test_no_user_message_does_not_create_profile(self) -> None:
        chat = ExtractedChat(
            messages=[ChatMessage(speaker="other", side="left", text="在吗")]
        )

        profile = learn_user_style(self.db, "owner", "friend", chat)

        self.assertIsNone(profile)
        self.assertEqual(self.db.query(UserStyleProfile).count(), 0)

    def test_profiles_are_isolated_by_user_and_relation(self) -> None:
        learn_user_style(
            self.db,
            "owner",
            "friend",
            ExtractedChat(
                messages=[ChatMessage(speaker="me", side="right", text="朋友口吻")]
            ),
        )
        learn_user_style(
            self.db,
            "owner",
            "family",
            ExtractedChat(
                messages=[ChatMessage(speaker="me", side="right", text="家人说法")]
            ),
        )
        learn_user_style(
            self.db,
            "other-owner",
            "friend",
            ExtractedChat(
                messages=[ChatMessage(speaker="me", side="right", text="另一个账号")]
            ),
        )

        friend = get_saved_style_snapshot(self.db, "owner", "friend")

        self.assertEqual(friend.phrases, ("朋友口吻",))
        self.assertNotIn("家人说法", friend.phrases)
        self.assertNotIn("另一个账号", friend.phrases)

    def test_reads_legacy_prefixed_phrases_without_prefix(self) -> None:
        self.db.add(
            UserStyleProfile(
                user_id="owner",
                relation_type="friend",
                style_summary="旧画像",
                example_phrases="我: 行啊\n我 (right): 晚点说",
            )
        )
        self.db.commit()

        snapshot = get_saved_style_snapshot(self.db, "owner", "friend")

        self.assertEqual(snapshot.phrases, ("行啊", "晚点说"))
