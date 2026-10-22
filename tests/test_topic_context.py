import json
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fixtures_v021 import GENERATION
from chat_reply_agent.database import Base, get_db
from chat_reply_agent.main import app, _build_reply_response, _normalize_confirmed_chat
from chat_reply_agent.models import ContactMemory, MemoryEpisode, MemoryJob
from chat_reply_agent.schemas import ChatMessage, ExtractedChat
from chat_reply_agent.services.context_service import build_context_retrieval_query, build_model_chat_context
from chat_reply_agent.services.evaluation_service import load_evaluation_dataset, run_evaluation_case, DEFAULT_DATASET_PATH
from chat_reply_agent.services.generation_service import understand_and_generate
from chat_reply_agent.services.llm_client import ModelCallError
from chat_reply_agent.services.memory_queue_service import process_next_memory_job
from chat_reply_agent.services.vision_service import _parse_extracted_chat, merge_extracted_chats


EXPECTED_TOPIC = "已有安排无法应邀，提出六点半短暂补位后对方表示不用"
STALE_TOPIC = "去玩了"


def invitation_case():
    return next(case for case in load_evaluation_dataset()[2] if case.id == "friend_withdrawn_invitation")


def invitation_generation(topic=EXPECTED_TOPIC):
    payload = json.loads(GENERATION)
    payload["topic"] = topic
    payload["analysis"].update(
        scene="邀约与有条件补位", emotion="可能体谅，也可能失落，无法确定",
        key_points=["已有安排无法参加", "只提出六点半后临时补位", "对方表示不用"],
    )
    return json.dumps(payload, ensure_ascii=False)


class TopicEvidenceTests(TestCase):
    def test_old_metadata_never_becomes_message_evidence_or_retrieval_query(self):
        chat = invitation_case().expected_chat()
        chat.topic, chat.topic_source = STALE_TOPIC, "context"
        chat.last_message_from_other = "旧的最后一条"
        chat.notes = ["未经证实的情绪推断"]
        for text in (chat.to_text(), build_model_chat_context(chat), build_context_retrieval_query(chat)):
            for stale in (STALE_TOPIC, "旧的最后一条", "未经证实的情绪推断"):
                self.assertNotIn(stale, text)
            self.assertIn("算了算了你去玩", text)
            self.assertIn("六点半", text)

    def test_vision_ignores_unsolicited_topic_and_merge_or_confirmation_invalidates_it(self):
        raw = json.dumps({"topic": STALE_TOPIC, "messages": [
            {"side": "left", "text": "算了算了你去玩", "text_confidence": 0.95, "position_confidence": 0.95},
        ]}, ensure_ascii=False)
        extracted = _parse_extracted_chat(raw)
        self.assertEqual((extracted.topic, extracted.topic_source), ("", "pending"))
        extracted.topic, extracted.topic_source = STALE_TOPIC, "context"
        for result in (merge_extracted_chats([extracted]), _normalize_confirmed_chat(extracted, "已校对")):
            self.assertEqual((result.topic, result.topic_source), ("", "pending"))
            self.assertEqual(result.last_message_from_other, "算了算了你去玩")
        self.assertEqual(extracted.topic, STALE_TOPIC)

    def test_topic_and_replies_share_one_call_with_full_context_and_fact_boundaries(self):
        case = invitation_case()
        chat_text = build_model_chat_context(case.expected_chat(), case.background_context)
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=invitation_generation()) as model:
            result = understand_and_generate(chat_text, {}, "", "")
        model.assert_called_once()
        prompt = model.call_args.args[1]
        self.assertIn(case.background_context, prompt)
        for message in case.messages:
            self.assertIn(message.text, prompt)
        for rule in ("事实、意向、条件方案", "不是最后一句的关键词", "直接矛盾", "不包含无法证实的情绪"):
            self.assertIn(rule, prompt)
        self.assertIn("topic", model.call_args.kwargs["response_schema"]["required"])
        self.assertEqual(result.topic, EXPECTED_TOPIC)

    def test_unknown_topic_can_be_empty_but_missing_or_wrong_type_fails(self):
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=invitation_generation("   ")):
            self.assertEqual(understand_and_generate("对方：嗯", {}, "", "").topic, "")
        for value in (None, [], 3, "字" * 121):
            with self.subTest(value=value), patch("chat_reply_agent.services.generation_service.call_text_model", return_value=invitation_generation(value)):
                with self.assertRaises(ModelCallError):
                    understand_and_generate("chat", {}, "", "")
        payload = json.loads(GENERATION)
        del payload["topic"]
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=json.dumps(payload)):
            with self.assertRaises(ModelCallError):
                understand_and_generate("chat", {}, "", "")

    def test_evaluation_uses_the_same_background_and_message_evidence(self):
        case = invitation_case()
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=invitation_generation()) as model, patch(
                "chat_reply_agent.services.quality_service.call_text_model", return_value=None):
            result = run_evaluation_case(case, DEFAULT_DATASET_PATH)
        self.assertIsNone(result.error)
        model.assert_called_once()
        self.assertIn(case.background_context, model.call_args.args[1])


class TopicPipelineTests(TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine)
        self.db = self.factory()
        app.dependency_overrides[get_db] = self.session_override
        self.client = TestClient(app)

    def session_override(self):
        with self.factory() as db:
            yield db

    def tearDown(self):
        app.dependency_overrides.pop(get_db, None)
        self.client.close()
        self.db.close()
        self.engine.dispose()

    def test_new_topic_reaches_response_and_background_memory_without_extra_generation(self):
        case = invitation_case()
        chat = case.expected_chat()
        chat.topic = STALE_TOPIC
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=invitation_generation()) as model, patch(
                "chat_reply_agent.services.quality_service.call_text_model", return_value=None), patch(
                "chat_reply_agent.main.retrieve_relevant_memories", return_value=[]) as retrieval, patch(
                "chat_reply_agent.services.contact_memory_service.call_text_model") as memory_model:
            response = _build_reply_response(self.db, "owner", "friend", "contact", "朋友", chat,
                                             background_context=case.background_context)
        model.assert_called_once()
        memory_model.assert_not_called()
        self.assertNotIn(STALE_TOPIC, model.call_args.args[1])
        self.assertNotIn(STALE_TOPIC, retrieval.call_args.kwargs["query_text"])
        self.assertEqual(response.extracted_chat.topic, EXPECTED_TOPIC)
        self.assertEqual(response.extracted_chat.topic_source, "context")
        self.assertEqual(chat.topic, STALE_TOPIC)
        self.assertNotIn(EXPECTED_TOPIC, response.screenshot_text)
        self.assertNotIn(case.background_context, response.screenshot_text)
        job = self.db.get(MemoryJob, response.memory_job_id)
        queued = json.loads(job.payload_json)
        self.assertEqual(queued["extracted_chat"]["topic"], EXPECTED_TOPIC)
        self.assertNotIn(STALE_TOPIC, job.payload_json)
        self.assertNotIn(EXPECTED_TOPIC, queued["chat_text"])
        self.assertIn(case.background_context, queued["chat_text"])
        memory_result = json.dumps({"summary": EXPECTED_TOPIC, "facts": [], "interaction_preferences": []}, ensure_ascii=False)
        with patch("chat_reply_agent.services.contact_memory_service.call_text_model", return_value=memory_result):
            self.assertTrue(process_next_memory_job(self.factory))
        self.db.expire_all()
        self.assertEqual(self.db.query(MemoryEpisode).one().topic, EXPECTED_TOPIC)
        self.assertEqual(self.db.query(ContactMemory).one().last_topic, EXPECTED_TOPIC)

    def test_corrected_and_confirmed_routes_discard_client_topic_even_when_marked_context(self):
        case = invitation_case()
        chat = case.expected_chat().model_dump()
        chat.update(topic=STALE_TOPIC, topic_source="context")
        for route in ("/api/reply/from-chat", "/api/reply/corrected"):
            with self.subTest(route=route), patch("chat_reply_agent.services.generation_service.call_text_model", return_value=invitation_generation()) as model, patch(
                    "chat_reply_agent.services.quality_service.call_text_model", return_value=None):
                response = self.client.post(route, json={"relation_type": "friend", "extracted_chat": chat,
                    "background_context": case.background_context, "save_to_memory": False})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["extracted_chat"]["topic"], EXPECTED_TOPIC)
            self.assertNotIn(STALE_TOPIC, model.call_args.args[1])
        self.assertEqual(self.db.query(MemoryJob).count(), 0)

    def test_background_only_uses_topic_but_fallback_or_uncertainty_never_restores_old_topic(self):
        for raw, expected in ((invitation_generation(), EXPECTED_TOPIC), (invitation_generation(""), ""), (None, "")):
            with self.subTest(expected=expected, configured=raw is not None), patch(
                    "chat_reply_agent.services.generation_service.call_text_model", return_value=raw), patch(
                    "chat_reply_agent.services.quality_service.call_text_model", return_value=None):
                response = self.client.post("/api/reply/from-chat", json={"relation_type": "friend",
                    "extracted_chat": {"messages": [], "topic": STALE_TOPIC, "topic_source": "context"},
                    "background_context": invitation_case().background_context, "save_to_memory": False})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["extracted_chat"]["topic"], expected)
            self.assertEqual(response.json()["extracted_chat"]["topic_source"], "context" if expected else "pending")
            self.assertEqual(response.json()["screenshot_text"], "")

    def test_invalid_topic_never_queues_memory(self):
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=invitation_generation(None)):
            response = self.client.post("/api/reply/from-chat", json={"relation_type": "friend",
                "extracted_chat": {"messages": []}, "background_context": invitation_case().background_context})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.db.query(MemoryJob).count(), 0)
