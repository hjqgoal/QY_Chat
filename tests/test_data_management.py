from contextlib import ExitStack
from unittest import TestCase
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from chat_reply_agent.database import Base
from chat_reply_agent.main import create_reply_from_correction
from chat_reply_agent.models import (
    ChatCorrection,
    ContactMemory,
    ConversationMemory,
    MemoryEpisode,
    ReplyFeedback,
    StyleExample,
    UserStyleProfile,
)
from chat_reply_agent.schemas import (
    ChatMessage,
    ContactMemoryUpdateRequest,
    CorrectedReplyRequest,
    ExtractedChat,
    ReplyControls,
)
from chat_reply_agent.services.contact_memory_service import (
    get_contact_memory,
    update_contact_memory_manually,
)
from chat_reply_agent.services.data_management_service import (
    delete_contact_memory_data,
    delete_memory_episode_data,
    reset_style_data,
)
from chat_reply_agent.services.style_example_service import (
    create_manual_style_example,
    delete_style_example,
    get_style_examples_summary,
)


class DataManagementTests(TestCase):
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

    def test_manual_contact_update_is_scoped_to_owner(self) -> None:
        owner = self._contact("owner", "xiaolin", "旧概要")
        other = self._contact("other", "xiaolin", "另一个账号的概要")
        owner_episode = self._episode("owner", "xiaolin", None)
        other_episode = self._episode("other", "xiaolin", None)
        self.db.add_all([owner, other, owner_episode, other_episode])
        self.db.commit()
        payload = ContactMemoryUpdateRequest(
            user_id="owner",
            contact_name="小林",
            relation_type="colleague",
            summary="确认周末一起吃火锅。",
            facts=["对方不吃辣", "对方不吃辣"],
            interaction_preferences=["约时间时希望直接一点"],
            last_topic="周末火锅",
            last_scenario="邀约",
        )

        snapshot = update_contact_memory_manually(
            self.db,
            user_id="owner",
            contact_id="xiaolin",
            payload=payload,
        )

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.summary, "确认周末一起吃火锅。")
        self.assertEqual(snapshot.facts, ["对方不吃辣"])
        self.assertEqual(snapshot.relation_type, "colleague")
        self.assertEqual(
            self.db.query(MemoryEpisode)
            .filter_by(user_id="owner", contact_id="xiaolin")
            .one()
            .relation_type,
            "colleague",
        )
        self.assertEqual(
            self.db.query(MemoryEpisode)
            .filter_by(user_id="other", contact_id="xiaolin")
            .one()
            .relation_type,
            "friend",
        )
        untouched = get_contact_memory(self.db, "other", "xiaolin")
        self.assertEqual(untouched.summary, "另一个账号的概要")

    def test_contact_delete_cascades_only_owned_memory(self) -> None:
        owner_contact = self._contact("owner", "xiaolin", "待删除")
        other_contact = self._contact("other", "xiaolin", "必须保留")
        owner_conversation = ConversationMemory(
            user_id="owner",
            relation_type="friend",
            summary="owner conversation",
        )
        other_conversation = ConversationMemory(
            user_id="other",
            relation_type="friend",
            summary="other conversation",
        )
        self.db.add_all(
            [owner_contact, other_contact, owner_conversation, other_conversation]
        )
        self.db.commit()
        self.db.add_all(
            [
                self._episode("owner", "xiaolin", owner_conversation.id),
                self._episode("other", "xiaolin", other_conversation.id),
            ]
        )
        self.db.commit()

        result = delete_contact_memory_data(
            self.db,
            user_id="owner",
            contact_id="xiaolin",
        )

        self.assertEqual(result.deleted_contact_memory, 1)
        self.assertEqual(result.deleted_memory_episodes, 1)
        self.assertEqual(result.deleted_conversation_memories, 1)
        self.assertIsNone(get_contact_memory(self.db, "owner", "xiaolin"))
        self.assertIsNotNone(get_contact_memory(self.db, "other", "xiaolin"))
        self.assertEqual(
            self.db.query(MemoryEpisode).filter_by(user_id="other").count(),
            1,
        )
        self.assertEqual(
            self.db.query(ConversationMemory).filter_by(user_id="other").count(),
            1,
        )

    def test_single_episode_delete_checks_owner_and_removes_source_memory(self) -> None:
        conversation = ConversationMemory(
            user_id="owner",
            relation_type="friend",
            summary="关联概要",
        )
        self.db.add(conversation)
        self.db.commit()
        episode = self._episode("owner", "xiaolin", conversation.id)
        self.db.add(episode)
        self.db.commit()

        denied = delete_memory_episode_data(
            self.db,
            user_id="other",
            contact_id="xiaolin",
            memory_id=episode.id,
        )
        deleted = delete_memory_episode_data(
            self.db,
            user_id="owner",
            contact_id="xiaolin",
            memory_id=episode.id,
        )

        self.assertFalse(denied)
        self.assertTrue(deleted)
        self.assertEqual(self.db.query(MemoryEpisode).count(), 0)
        self.assertEqual(self.db.query(ConversationMemory).count(), 0)

    def test_manual_style_examples_can_be_deduplicated_and_deleted(self) -> None:
        first = create_manual_style_example(
            self.db,
            user_id="owner",
            relation_type="friend",
            text="  行，那先这样  ",
        )
        duplicate = create_manual_style_example(
            self.db,
            user_id="owner",
            relation_type="friend",
            text="行，那先这样",
        )

        self.assertEqual(first.id, duplicate.id)
        self.assertIn(
            "行，那先这样",
            get_style_examples_summary(self.db, "owner", "friend"),
        )
        self.assertFalse(
            delete_style_example(self.db, "other", "friend", first.id)
        )
        self.assertTrue(
            delete_style_example(self.db, "owner", "friend", first.id)
        )

    def test_deleting_feedback_style_example_removes_its_learning_record(self) -> None:
        feedback = ReplyFeedback(
            user_id="owner",
            relation_type="friend",
            feedback_type="accepted",
            candidate_text="那先这样",
        )
        self.db.add(feedback)
        self.db.commit()
        example = StyleExample(
            user_id="owner",
            relation_type="friend",
            text="那先这样",
            source="accepted",
            feedback_id=feedback.id,
        )
        self.db.add(example)
        self.db.commit()

        deleted = delete_style_example(
            self.db,
            user_id="owner",
            relation_type="friend",
            example_id=example.id,
        )

        self.assertTrue(deleted)
        self.assertEqual(self.db.query(StyleExample).count(), 0)
        self.assertEqual(self.db.query(ReplyFeedback).count(), 0)

    def test_style_reset_is_scoped_to_relation_and_user(self) -> None:
        self.db.add_all(
            [
                UserStyleProfile(
                    user_id="owner",
                    relation_type="friend",
                    style_summary="轻松",
                ),
                UserStyleProfile(
                    user_id="owner",
                    relation_type="family",
                    style_summary="家庭风格",
                ),
                StyleExample(
                    user_id="owner",
                    relation_type="friend",
                    text="行啊",
                    source="manual",
                ),
                ReplyFeedback(
                    user_id="owner",
                    relation_type="friend",
                    feedback_type="too_ai",
                ),
                StyleExample(
                    user_id="other",
                    relation_type="friend",
                    text="另一个账号",
                    source="manual",
                ),
            ]
        )
        self.db.commit()

        result = reset_style_data(
            self.db,
            user_id="owner",
            relation_type="friend",
        )

        self.assertEqual(result.deleted_profiles, 1)
        self.assertEqual(result.deleted_examples, 1)
        self.assertEqual(result.deleted_feedback, 1)
        self.assertEqual(
            self.db.query(UserStyleProfile).filter_by(relation_type="family").count(),
            1,
        )
        self.assertEqual(
            self.db.query(StyleExample).filter_by(user_id="other").count(),
            1,
        )

    def test_temporary_mode_writes_no_chat_or_style_data(self) -> None:
        payload = CorrectedReplyRequest(
            relation_type="friend",
            user_id="private-user",
            contact_id="xiaolin",
            contact_name="小林",
            extracted_chat=ExtractedChat(
                messages=[
                    ChatMessage(speaker="other", side="left", text="今晚出来吗"),
                    ChatMessage(speaker="me", side="right", text="我看看"),
                ],
                topic="今晚邀约",
                last_message_from_other="今晚出来吗",
                confidence=1,
            ),
            controls=ReplyControls(intent="decline"),
            save_to_memory=False,
        )
        model_targets = (
            "chat_reply_agent.services.analysis_service.call_text_model",
            "chat_reply_agent.services.scenario_service.call_text_model",
            "chat_reply_agent.services.reply_service.call_text_model",
            "chat_reply_agent.services.quality_service.call_text_model",
            "chat_reply_agent.services.contact_memory_service.call_text_model",
        )
        with ExitStack() as stack:
            for target in model_targets:
                stack.enter_context(patch(target, return_value=None))
            response = create_reply_from_correction(payload, self.db)

        self.assertFalse(response.memory_persisted)
        self.assertIsNone(response.saved_memory_id)
        self.assertIsNone(response.saved_correction_id)
        self.assertEqual(response.style_evidence.current_user_message_count, 1)
        self.assertEqual(response.style_evidence.current_examples, ["我看看"])
        self.assertEqual(response.style_evidence.profile_examples, [])
        self.assertTrue(response.style_evidence.current_examples)
        self.assertIsNone(response.memory_job_id)
        for model in (
            ChatCorrection,
            ContactMemory,
            ConversationMemory,
            MemoryEpisode,
            UserStyleProfile,
            StyleExample,
            ReplyFeedback,
        ):
            self.assertEqual(self.db.query(model).count(), 0, model.__name__)

    @staticmethod
    def _contact(user_id: str, contact_id: str, summary: str) -> ContactMemory:
        return ContactMemory(
            user_id=user_id,
            contact_id=contact_id,
            contact_name="小林",
            relation_type="friend",
            summary=summary,
        )

    @staticmethod
    def _episode(
        user_id: str,
        contact_id: str,
        conversation_memory_id: int | None,
    ) -> MemoryEpisode:
        return MemoryEpisode(
            user_id=user_id,
            contact_id=contact_id,
            relation_type="friend",
            topic="周末火锅",
            scenario_type="invitation",
            scenario_label="邀约",
            summary="聊过周末吃火锅。",
            chat_text="对方：周末吃火锅吗",
            conversation_memory_id=conversation_memory_id,
        )
