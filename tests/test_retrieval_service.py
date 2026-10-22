from contextlib import ExitStack
from datetime import UTC, datetime, timedelta, timezone
import json
from unittest import TestCase
from unittest.mock import patch
from fixtures_v021 import GENERATION, MEMORY
from chat_reply_agent.services.memory_queue_service import process_next_memory_job

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from chat_reply_agent.config import Settings
from chat_reply_agent.database import Base
from chat_reply_agent.main import _build_reply_response
from chat_reply_agent.schemas import ChatMessage, ChatScenario, ExtractedChat
from chat_reply_agent.services.retrieval_service import (
    _time_decay_factor,
    build_retrieval_context,
    list_memory_episodes,
    retrieve_relevant_memories,
    save_memory_episode,
)


class RetrievalServiceTests(TestCase):
    def setUp(self) -> None:
        self.settings = Settings(_env_file=None, memory_similarity_backend="lexical")
        settings_patch = patch(
            "chat_reply_agent.services.retrieval_service.get_settings",
            return_value=self.settings,
        )
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        session_factory = sessionmaker(bind=self.engine)
        self.db = session_factory()
        self.daily = ChatScenario(
            type="daily_chat",
            label="日常聊天",
            confidence=1,
        )
        self.invitation = ChatScenario(
            type="invitation",
            label="邀约",
            confidence=1,
        )

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_retrieval_ranks_relevant_episode_and_isolates_contact(self) -> None:
        dinner = self._save(
            contact_id="xiaolin",
            topic="周末吃火锅",
            summary="之前约好周末去吃火锅，还没有确定具体时间。",
            chat_text="对方: 周末一起去吃火锅吗\n我: 可以啊",
            scenario=self.invitation,
        )
        self._save(
            contact_id="xiaolin",
            topic="游戏充值",
            summary="讨论国际服充值和联机安排。",
            chat_text="对方: 国际服还能充值吗\n我: 我看看",
        )
        self._save(
            user_id="another-user",
            contact_id="xiaolin",
            topic="周末吃火锅",
            summary="另一个用户的联系人记忆，不得被召回。",
            chat_text="对方: 周末火锅",
            scenario=self.invitation,
        )
        self._save(
            contact_id="colleague-a",
            topic="周末吃火锅",
            summary="另一个联系人的记忆，不得被召回。",
            chat_text="对方: 周末火锅",
            scenario=self.invitation,
        )

        results = retrieve_relevant_memories(
            self.db,
            user_id="default",
            contact_id="xiaolin",
            query_text="周末火锅还去吗",
            relation_type="friend",
            scenario_type="invitation",
        )

        self.assertTrue(results)
        self.assertEqual(results[0].id, dinner.id)
        self.assertGreaterEqual(results[0].relevance_score, 10)
        self.assertNotIn("另一个", "".join(item.summary for item in results))

    def test_exact_current_chat_is_not_retrieved(self) -> None:
        current_chat = "对方: 周末还去吃火锅吗\n我: 去啊"
        duplicate = self._save(
            contact_id="xiaolin",
            topic="周末火锅",
            summary="已经聊过周末吃火锅。",
            chat_text=current_chat,
            scenario=self.invitation,
        )
        older = self._save(
            contact_id="xiaolin",
            topic="火锅口味",
            summary="对方更喜欢清汤锅。",
            chat_text="对方: 我还是更喜欢清汤锅",
            scenario=self.daily,
        )

        results = retrieve_relevant_memories(
            self.db,
            user_id="default",
            contact_id="xiaolin",
            query_text="周末火锅清汤锅",
            relation_type="friend",
            scenario_type="invitation",
            exclude_chat_text=current_chat,
        )

        self.assertNotIn(duplicate.id, [item.id for item in results])
        self.assertIn(older.id, [item.id for item in results])

    def test_confirmation_notes_do_not_break_current_chat_exclusion(self) -> None:
        stored_chat = (
            "对方 (left): 你到哪了\n"
            "我 (right): 快了\n"
            "对话主题：见面安排\n"
            "对方最后一条消息：你到哪了\n"
            "识别备注：用户已确认截图识别内容。"
        )
        episode = self._save(
            contact_id="xiaolin",
            topic="见面安排",
            summary="正在确认见面时间。",
            chat_text=stored_chat,
        )

        results = retrieve_relevant_memories(
            self.db,
            user_id="default",
            contact_id="xiaolin",
            query_text="你到哪了\n快了\n见面安排",
            exclude_chat_text="你到哪了\n快了\n见面安排",
        )

        self.assertNotIn(episode.id, [item.id for item in results])

    def test_recent_list_and_context_are_explainable(self) -> None:
        first = self._save(
            contact_id="xiaolin",
            topic="看电影",
            summary="聊过周五去看电影。",
            chat_text="对方: 周五看电影吗",
        )
        first.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=3)
        self.db.commit()
        second = self._save(
            contact_id="xiaolin",
            topic="吃饭",
            summary="刚聊过晚上吃饭。",
            chat_text="对方: 晚上吃什么",
        )

        recent = list_memory_episodes(
            self.db,
            user_id="default",
            contact_id="xiaolin",
        )
        context = build_retrieval_context("稳定事实：对方不吃辣。", recent[:1])

        self.assertEqual([item.id for item in recent], [second.id, first.id])
        self.assertEqual(recent[0].similarity_method, "not_scored")
        self.assertIsNone(recent[0].time_decay_factor)
        self.assertIn("只作事实参考", context)
        self.assertIn("以当前聊天为准", context)
        self.assertIn("对方不吃辣", context)

    def test_semantic_similarity_is_multiplied_by_age_without_bonuses(self) -> None:
        now = datetime(2026, 9, 17, tzinfo=UTC)
        older = self._save(
            contact_id="xiaolin", topic="old", summary="old",
            chat_text="old", scenario=self.invitation,
        )
        newer = self._save(
            contact_id="xiaolin", topic="new", summary="new", chat_text="new",
        )
        older.created_at = (now - timedelta(days=30)).replace(tzinfo=None)
        newer.created_at = now.replace(tzinfo=None)
        self.db.commit()
        self.settings.memory_similarity_backend = "semantic"
        with patch(
            "chat_reply_agent.services.retrieval_service.semantic_similarities",
            return_value=[0.6, 0.8],
        ), patch("chat_reply_agent.services.retrieval_service.datetime") as clock:
            clock.now.return_value = now
            results = retrieve_relevant_memories(
                self.db, user_id="default", contact_id="xiaolin",
                query_text="no lexical overlap", relation_type="friend",
                scenario_type="invitation",
            )
        self.assertEqual([item.id for item in results], [newer.id, older.id])
        self.assertEqual([item.relevance_score for item in results], [60, 40])
        self.assertEqual(results[1].similarity_score, 0.8)
        self.assertEqual(results[1].time_decay_factor, 0.5)
        self.assertEqual(results[1].decay_half_life_days, 30)
        self.assertEqual(results[1].similarity_method, "semantic")

    def test_scenario_override_and_global_fallback(self) -> None:
        now = datetime(2026, 9, 17, tzinfo=UTC)
        slow = self._save(
            contact_id="xiaolin", topic="slow", summary="slow", chat_text="slow",
        )
        fast = self._save(
            contact_id="xiaolin", topic="fast", summary="fast", chat_text="fast",
            scenario=self.invitation,
        )
        for episode in (slow, fast):
            episode.created_at = (now - timedelta(days=30)).replace(tzinfo=None)
        self.db.commit()
        self.settings.memory_similarity_backend = "semantic"
        self.settings.memory_decay_half_life_days = 60
        self.settings.memory_decay_half_life_by_scenario = {"invitation": 30}
        with patch(
            "chat_reply_agent.services.retrieval_service.semantic_similarities",
            return_value=[0.8, 0.8],
        ), patch("chat_reply_agent.services.retrieval_service.datetime") as clock:
            clock.now.return_value = now
            results = retrieve_relevant_memories(
                self.db, user_id="default", contact_id="xiaolin", query_text="query",
            )
        self.assertEqual([item.id for item in results], [slow.id, fast.id])
        self.assertEqual([item.decay_half_life_days for item in results], [60, 30])
        self.assertEqual([item.relevance_score for item in results], [57, 40])

    def test_raw_scores_determine_ranking_and_threshold_before_rounding(self) -> None:
        now = datetime(2026, 9, 17, tzinfo=UTC)
        episodes = [
            self._save(contact_id="xiaolin", topic=str(i), summary=str(i), chat_text=str(i))
            for i in range(3)
        ]
        for episode in episodes:
            episode.created_at = now.replace(tzinfo=None)
        self.db.commit()
        self.settings.memory_similarity_backend = "semantic"
        with patch(
            "chat_reply_agent.services.retrieval_service.semantic_similarities",
            return_value=[0.1599, 0.801, 0.804],
        ), patch("chat_reply_agent.services.retrieval_service.datetime") as clock:
            clock.now.return_value = now
            results = retrieve_relevant_memories(
                self.db, user_id="default", contact_id="xiaolin", query_text="query",
            )
        self.assertEqual([item.id for item in results], [episodes[0].id, episodes[1].id])
        self.assertEqual([item.relevance_score for item in results], [80, 80])

    def test_missing_semantic_model_is_explicit_lexical_fallback(self) -> None:
        self._save(
            contact_id="xiaolin", topic="火锅", summary="火锅", chat_text="火锅",
        )
        self.settings.memory_similarity_backend = "semantic"
        with patch(
            "chat_reply_agent.services.retrieval_service.semantic_similarities",
            return_value=None,
        ):
            results = retrieve_relevant_memories(
                self.db, user_id="default", contact_id="xiaolin", query_text="火锅",
            )
        self.assertTrue(results)
        self.assertEqual(results[0].similarity_method, "lexical_fallback")
        self.assertEqual(results[0].relevance_score, round(
            results[0].similarity_score * results[0].time_decay_factor * 100,
        ))

    def test_stale_memory_is_not_deleted_and_unrelated_new_memory_is_not_boosted(self) -> None:
        stale = self._save(
            contact_id="xiaolin", topic="火锅", summary="火锅", chat_text="火锅",
        )
        stale.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=365)
        self.db.commit()
        self._save(
            contact_id="xiaolin", topic="abcd", summary="abcd", chat_text="abcd",
        )
        results = retrieve_relevant_memories(
            self.db, user_id="default", contact_id="xiaolin", query_text="火锅",
            relation_type="friend", scenario_type="daily_chat",
        )
        self.assertEqual(results, [])
        self.assertEqual(len(list_memory_episodes(
            self.db, user_id="default", contact_id="xiaolin",
        )), 2)

    def test_short_followup_retrieves_without_an_appended_generated_topic(self) -> None:
        episode = self._save(contact_id="xiaolin", topic="周末火锅",
            summary="约好周末一起吃火锅", chat_text="对方: 周末一起吃火锅吗\n我: 可以啊")
        results = retrieve_relevant_memories(self.db, user_id="default", contact_id="xiaolin",
                                            query_text="火锅几点去呀")
        self.assertEqual([item.id for item in results], [episode.id])
        unrelated = retrieve_relevant_memories(self.db, user_id="default", contact_id="xiaolin",
                                              query_text="配送通知已经收到")
        self.assertEqual(unrelated, [])

    def test_incidental_shared_phrase_does_not_make_different_topics_relevant(self) -> None:
        self._save(contact_id="xiaolin", topic="项目进度",
                   summary="收到项目进度报告，稍后回复", chat_text="对方: 收到报告了\n我: 好")
        results = retrieve_relevant_memories(self.db, user_id="default", contact_id="xiaolin",
                                            query_text="刚才收到新的配送通知，快递将在明天送达")
        self.assertEqual(results, [])

    def test_reply_pipeline_saves_then_retrieves_related_episode(self) -> None:
        first_chat = ExtractedChat(
            messages=[
                ChatMessage(speaker="other", side="left", text="周末一起吃火锅吗"),
                ChatMessage(speaker="me", side="right", text="可以啊"),
            ],
            topic="周末火锅",
            last_message_from_other="周末一起吃火锅吗",
            confidence=1,
        )
        second_chat = ExtractedChat(
            messages=[
                ChatMessage(speaker="other", side="left", text="火锅几点去呀"),
            ],
            topic="确认火锅时间",
            last_message_from_other="火锅几点去呀",
            confidence=1,
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
            generation = json.loads(GENERATION)
            generation["topic"] = "周末火锅"
            generation["analysis"]["key_points"] = ["约好周末一起吃火锅"]
            stack.enter_context(patch("chat_reply_agent.services.generation_service.call_text_model",
                                      return_value=json.dumps(generation, ensure_ascii=False)))
            first = _build_reply_response(
                db=self.db,
                user_id="pipeline-user",
                relation_type="friend",
                contact_id="xiaolin",
                contact_name="小林",
                extracted_chat=first_chat,
            )
            with patch("chat_reply_agent.services.contact_memory_service.call_text_model", return_value=MEMORY):
                process_next_memory_job(sessionmaker(bind=self.engine))
            self.db.expire_all()
            second = _build_reply_response(
                db=self.db,
                user_id="pipeline-user",
                relation_type="friend",
                contact_id="xiaolin",
                contact_name="小林",
                extracted_chat=second_chat,
            )

        self.assertEqual(first.retrieved_memories, [])
        self.assertFalse(first.memory_persisted)
        self.assertEqual(first.memory_status, "pending")
        stages = [item["stage"] for item in first.diagnostics["stages"]]
        for stage in ("memory.contact", "memory.summary", "memory.episode", "style.learn"):
            self.assertNotIn(stage, stages)
        self.assertIn("memory.enqueue", stages)
        self.assertNotEqual(first.diagnostics["request_id"], second.diagnostics["request_id"])
        self.assertEqual(first.style_evidence.current_examples, ["可以啊"])
        self.assertEqual(first.style_evidence.profile_examples, [])
        self.assertTrue(second.retrieved_memories)
        self.assertEqual(second.retrieved_memories[0].topic, "周末火锅")
        self.assertIn("火锅", second.retrieved_memories[0].summary)
        self.assertEqual(second.style_evidence.current_user_message_count, 0)
        self.assertIn("可以啊", second.style_evidence.profile_examples)
        self.assertTrue(second.strategy)

    def test_dashscope_routing_preserves_contact_scope_and_fallback_label(self):
        self._save(contact_id="xiaolin", topic="周末火锅", summary="周末去吃火锅", chat_text="周末火锅")
        self._save(contact_id="another", topic="私有话题", summary="其他联系人", chat_text="不应发送")
        self.settings.memory_similarity_backend = "semantic"
        self.settings.memory_embedding_provider = "dashscope"
        self.settings.memory_embedding_model = "text-embedding-v4"
        self.settings.openai_api_key = "test-shared-key"
        self.settings.memory_embedding_reuse_chat_credentials = True
        self.settings.openai_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        with patch("chat_reply_agent.services.retrieval_service.semantic_similarities", return_value=[0.9]) as embedding:
            results = retrieve_relevant_memories(self.db, user_id="default", contact_id="xiaolin", query_text="周末火锅")
        self.assertEqual(results[0].similarity_method, "semantic")
        self.assertEqual(embedding.call_args.kwargs["provider"], "dashscope")
        self.assertEqual(embedding.call_args.kwargs["api_key"], "test-shared-key")
        self.assertEqual(len(embedding.call_args.args[1]), 1)
        self.assertNotIn("其他联系人", embedding.call_args.args[1][0])
        self.settings.memory_embedding_api_key = "test-embedding-key"
        with patch("chat_reply_agent.services.retrieval_service.semantic_similarities", return_value=None) as embedding:
            results = retrieve_relevant_memories(self.db, user_id="default", contact_id="xiaolin", query_text="周末火锅")
        self.assertTrue(results)
        self.assertEqual(results[0].similarity_method, "lexical_fallback")
        self.assertEqual(embedding.call_args.kwargs["api_key"], "test-embedding-key")

    def _save(
        self,
        *,
        contact_id: str,
        topic: str,
        summary: str,
        chat_text: str,
        scenario: ChatScenario | None = None,
        user_id: str = "default",
    ):
        return save_memory_episode(
            self.db,
            user_id=user_id,
            contact_id=contact_id,
            relation_type="friend",
            topic=topic,
            scenario=scenario or self.daily,
            summary=summary,
            chat_text=chat_text,
        )


class TimeDecayTests(TestCase):
    def test_half_life_and_monotonic_decay(self) -> None:
        now = datetime(2026, 9, 17, tzinfo=UTC)
        factors = [
            _time_decay_factor(now - timedelta(days=days), 30, now=now)
            for days in (0, 15, 30, 60, 365)
        ]
        self.assertEqual(factors[0], 1)
        self.assertEqual(factors[2], 0.5)
        self.assertEqual(factors[3], 0.25)
        self.assertEqual(factors, sorted(factors, reverse=True))

    def test_timezones_naive_utc_and_future_dates(self) -> None:
        now = datetime(2026, 9, 17, tzinfo=UTC)
        created = now - timedelta(days=30)
        for timestamp in (created, created.replace(tzinfo=None),
                          created.astimezone(timezone(timedelta(hours=8)))):
            self.assertEqual(_time_decay_factor(timestamp, 30, now=now), 0.5)
        self.assertEqual(_time_decay_factor(now + timedelta(days=1), 30, now=now), 1)

    def test_invalid_half_life_is_rejected(self) -> None:
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _time_decay_factor(datetime.now(UTC), value)
