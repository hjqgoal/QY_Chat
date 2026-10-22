import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from chat_reply_agent.config import Settings
from chat_reply_agent.database import Base, get_db
from chat_reply_agent.main import app, _build_reply_response
from chat_reply_agent.schemas import ChatScenario, ExtractedChat, MemoryRetrievalDiagnostics
from chat_reply_agent.services.embedding_service import embedding_options, embedding_runtime_status, semantic_similarities
from chat_reply_agent.services.llm_client import ModelCallError
from chat_reply_agent.services.retrieval_service import save_memory_episode, retrieve_relevant_memories
from fixtures_v021 import GENERATION
from scripts.check_embedding_connection import run_probe, QUERY, RELEVANT


def api_settings(**overrides):
    values = dict(_env_file=None, memory_similarity_backend="semantic", memory_embedding_provider="dashscope",
                  memory_embedding_model="text-embedding-v4", memory_embedding_api_key="embedding-test-secret",
                  memory_embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                  memory_embedding_fallback="error")
    values.update(overrides)
    return Settings(**values)


def fake_embeddings(**kwargs):
    rows = []
    for index, text in enumerate(kwargs["input"]):
        vector = [1.0, 0.0] if "不刺激" in text else [0.99, 0.01] if "清淡" in text else [0.0, 1.0]
        rows.append(SimpleNamespace(index=index, embedding=vector))
    return SimpleNamespace(data=list(reversed(rows)))


class EmbeddingConfigurationTests(TestCase):
    def test_dedicated_credentials_are_used_without_inheriting_chat_secret(self):
        settings = api_settings(openai_api_key="chat-test-secret")
        self.assertEqual(embedding_options(settings)["api_key"], "embedding-test-secret")
        settings.memory_embedding_api_key = ""
        with self.assertRaises(ModelCallError):
            embedding_options(settings)
        self.assertEqual(embedding_runtime_status(settings)["readiness"], "configuration_missing")
        self.assertNotIn("chat-test-secret", str(embedding_runtime_status(settings)))

    def test_opt_in_reuse_only_allows_exact_same_endpoint(self):
        settings = api_settings(memory_embedding_api_key="", memory_embedding_reuse_chat_credentials=True,
                                openai_api_key="chat-test-secret", openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/")
        self.assertEqual(embedding_options(settings)["api_key"], "chat-test-secret")
        settings.memory_embedding_base_url = "https://another-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        with self.assertRaises(ModelCallError) as error:
            embedding_options(settings)
        self.assertNotIn("chat-test-secret", str(error.exception))

    def test_compatible_api_uses_explicit_key_and_validates_https_base(self):
        settings = api_settings(memory_embedding_provider="openai_compatible", memory_embedding_base_url="https://embedding.example.test/v1")
        self.assertEqual(embedding_options(settings)["provider"], "openai_compatible")
        for url in ("http://embedding.example.test/v1", "https://user:secret@embedding.example.test/v1",
                    "https://embedding.example.test/v1?token=secret", "https://embedding.example.test/v1/embeddings",
                    "https://[bad-address", "https://embedding.example.test:invalid/v1",
                    "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"):
            settings.memory_embedding_base_url = url
            with self.subTest(url=url), self.assertRaises(ModelCallError):
                embedding_options(settings)

    def test_local_mode_does_not_need_api_credentials(self):
        options = embedding_options(api_settings(memory_embedding_provider="local", memory_embedding_api_key=""))
        self.assertIsNone(options["api_key"])
        self.assertEqual(options["base_url"], "")

    def test_strict_provider_errors_do_not_expose_secret_or_chat(self):
        client = Mock()
        client.embeddings.create.side_effect = RuntimeError("embedding-test-secret private-chat-text")
        with patch("chat_reply_agent.services.embedding_service.OpenAI", return_value=client), self.assertRaises(ModelCallError) as error:
            semantic_similarities("q", ["d"], **embedding_options(api_settings()), raise_on_error=True)
        self.assertNotIn("embedding-test-secret", str(error.exception))
        self.assertNotIn("private-chat-text", str(error.exception))
        client.close.assert_called_once()

    def test_compatible_provider_makes_embeddings_request_and_closes_client(self):
        client = Mock()
        client.embeddings.create.side_effect = fake_embeddings
        settings = api_settings(memory_embedding_provider="openai_compatible", memory_embedding_base_url="https://embedding.example.test/v1")
        with patch("chat_reply_agent.services.embedding_service.OpenAI", return_value=client) as factory:
            scores = semantic_similarities(QUERY, [RELEVANT], **embedding_options(settings), raise_on_error=True)
        self.assertGreater(scores[0], .9)
        self.assertEqual(factory.call_args.kwargs["api_key"], "embedding-test-secret")
        client.close.assert_called_once()


class SemanticRAGIntegrationTests(TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.settings = api_settings()
        self.client = Mock()
        self.client.embeddings.create.side_effect = fake_embeddings
        self.patches = [patch("chat_reply_agent.services.retrieval_service.get_settings", return_value=self.settings),
                        patch("chat_reply_agent.services.embedding_service.OpenAI", return_value=self.client)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.db.close()
        self.engine.dispose()

    def save(self, contact="demo", summary=RELEVANT):
        return save_memory_episode(self.db, user_id="test-user", contact_id=contact, relation_type="friend",
            topic="饮食偏好", summary=summary, chat_text=summary, scenario=ChatScenario())

    def test_probe_exercises_vectors_ranking_context_and_contact_isolation(self):
        result = run_probe(self.settings)
        self.assertTrue(result["semantic_rag_verified"])
        self.assertFalse(result["personal_database_used"])
        inputs = self.client.embeddings.create.call_args.kwargs["input"]
        self.assertEqual(len(inputs), 3)
        self.assertNotIn("另一位联系人", "".join(inputs))

    def test_strict_mode_does_not_replace_vector_failure_with_lexical_scores(self):
        self.save()
        self.client.embeddings.create.side_effect = TimeoutError("test")
        with self.assertRaises(ModelCallError), patch("chat_reply_agent.services.retrieval_service._lexical_similarities") as lexical:
            retrieve_relevant_memories(self.db, user_id="test-user", contact_id="demo", query_text="饮食偏好")
        lexical.assert_not_called()

    def test_no_history_makes_no_api_request_and_reports_not_scored(self):
        diagnostics = MemoryRetrievalDiagnostics()
        results = retrieve_relevant_memories(self.db, user_id="test-user", contact_id="missing", query_text=QUERY, diagnostics=diagnostics)
        self.assertEqual(results, [])
        self.assertEqual(diagnostics.method, "not_scored")
        self.client.embeddings.create.assert_not_called()

    def test_empty_results_still_report_the_real_method(self):
        self.save(summary="组队打游戏")
        diagnostics = MemoryRetrievalDiagnostics()
        results = retrieve_relevant_memories(self.db, user_id="test-user", contact_id="demo", query_text=QUERY, diagnostics=diagnostics)
        self.assertEqual(results, [])
        self.assertEqual(diagnostics.method, "semantic")
        self.assertEqual(diagnostics.candidate_count, 1)
        self.assertEqual(diagnostics.retrieved_count, 0)

    def test_explicit_fallback_is_labeled_even_without_hits(self):
        self.save(summary="组队打游戏")
        self.settings.memory_embedding_fallback = "lexical"
        self.settings.memory_embedding_api_key = ""
        diagnostics = MemoryRetrievalDiagnostics()
        retrieve_relevant_memories(self.db, user_id="test-user", contact_id="demo", query_text=QUERY, diagnostics=diagnostics)
        self.assertEqual(diagnostics.method, "lexical_fallback")
        self.assertIn("降级", diagnostics.message)
        self.client.embeddings.create.assert_not_called()

    def test_retrieved_memory_reaches_the_strong_generation_prompt(self):
        self.save()
        self.save(contact="another", summary="另一位联系人秘密")
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=GENERATION) as generation:
            result = _build_reply_response(db=self.db, user_id="test-user", contact_id="demo", contact_name="演示",
                relation_type="friend", extracted_chat=ExtractedChat(), background_context=QUERY, save_to_memory=False)
        self.assertEqual(result.retrieval.method, "semantic")
        self.assertIn(RELEVANT, generation.call_args.args[1])
        self.assertNotIn("另一位联系人秘密", generation.call_args.args[1])
        self.assertEqual(len(result.retrieved_memories), 1)

    def test_memory_search_endpoint_returns_clear_error_and_diagnostics(self):
        self.save()
        def db_override():
            yield self.db
        app.dependency_overrides[get_db] = db_override
        try:
            with TestClient(app, raise_server_exceptions=True) as client:
                response = client.get("/api/contacts/demo/memories", params={"user_id": "test-user", "query": QUERY})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["retrieval"]["method"], "semantic")
                self.settings.memory_embedding_api_key = ""
                response = client.get("/api/contacts/demo/memories", params={"user_id": "test-user", "query": QUERY})
                self.assertEqual(response.status_code, 502)
                self.assertIn("MEMORY_EMBEDDING_API_KEY", response.json()["detail"])
                with patch("chat_reply_agent.main.understand_and_generate") as generation, patch("chat_reply_agent.main.enqueue_memory") as memory:
                    response = client.post("/api/reply/from-chat", json={"user_id": "test-user", "contact_id": "demo",
                        "relation_type": "friend", "extracted_chat": {"messages": []}, "background_context": QUERY})
                self.assertEqual(response.status_code, 502)
                generation.assert_not_called()
                memory.assert_not_called()
                with patch("chat_reply_agent.main.enqueue_quality") as review:
                    response = client.post("/api/quality-jobs", json={"user_id": "test-user", "contact_id": "demo",
                        "relation_type": "friend", "extracted_chat": {"messages": []}, "background_context": QUERY,
                        "replies": json.loads(GENERATION)["replies"]})
                self.assertEqual(response.status_code, 502, response.text)
                review.assert_not_called()
        finally:
            app.dependency_overrides.pop(get_db, None)
