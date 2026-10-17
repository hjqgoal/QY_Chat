from types import SimpleNamespace
import math
from unittest import TestCase
from unittest.mock import Mock, patch

from chat_reply_agent.services.embedding_service import _load_model, semantic_similarities


class EmbeddingServiceTests(TestCase):
    def tearDown(self) -> None:
        _load_model.cache_clear()

    def test_local_only_model_is_loaded_once(self) -> None:
        _load_model.cache_clear()
        factory = Mock()
        with patch.dict("sys.modules", {
            "sentence_transformers": SimpleNamespace(SentenceTransformer=factory),
        }):
            self.assertIs(_load_model("local-model"), _load_model("local-model"))
        factory.assert_called_once_with(
            "local-model", device="cpu", local_files_only=True, trust_remote_code=False,
        )

    def test_missing_optional_package_falls_back_without_retrying_load(self) -> None:
        _load_model.cache_clear()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertLogs(level="WARNING") as logs:
                self.assertIsNone(_load_model("missing"))
                self.assertIsNone(_load_model("missing"))
        self.assertEqual(len(logs.output), 1)

    def test_normalized_cosine_similarity_is_batched_and_clipped(self) -> None:
        model = Mock()
        model.encode.return_value = [[1, 0], [0.8, 0.6], [-1, 0], [1.000001, 0]]
        with patch("chat_reply_agent.services.embedding_service._load_model", return_value=model):
            scores = semantic_similarities("query", ["a", "b", "c"], model_name="local")
        self.assertEqual(scores, [0.8, 0, 1])
        model.encode.assert_called_once_with(
            ["query", "a", "b", "c"], batch_size=32,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def test_invalid_vectors_and_encoder_errors_fall_back(self) -> None:
        for vectors in ([[1, 0]], [[1, 0], [1]], [[], []], [[1], [float("nan")]]):
            model = Mock()
            model.encode.return_value = vectors
            with self.subTest(vectors=vectors), patch(
                "chat_reply_agent.services.embedding_service._load_model", return_value=model,
            ), self.assertLogs(level="WARNING"):
                self.assertIsNone(semantic_similarities("q", ["a"], model_name="local"))
        model.encode.side_effect = RuntimeError("private text must not be logged")
        with patch(
            "chat_reply_agent.services.embedding_service._load_model", return_value=model,
        ), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(semantic_similarities("q", ["a"], model_name="local"))
        self.assertNotIn("private text", "".join(logs.output))

    def test_empty_documents_do_not_load_model(self) -> None:
        with patch("chat_reply_agent.services.embedding_service._load_model") as loader:
            self.assertEqual(semantic_similarities("q", [], model_name="local"), [])
        loader.assert_not_called()


class DashScopeEmbeddingTests(TestCase):
    def setUp(self):
        self.client = Mock()
        self.factory = patch(
            "chat_reply_agent.services.embedding_service.OpenAI", return_value=self.client,
        ).start()
        self.local = patch("chat_reply_agent.services.embedding_service._load_model").start()
        self.addCleanup(patch.stopall)

    def score(self, documents, **overrides):
        options = dict(model_name="text-embedding-v4", provider="dashscope",
                       api_key="test-only-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        options.update(overrides)
        return semantic_similarities("query", documents, **options)

    def test_sixty_candidates_are_batched_ordered_and_cosine_normalized(self):
        def respond(**kwargs):
            rows = []
            for index, text in enumerate(kwargs["input"]):
                vector = [3, 0] if text == "query" else [int(text.split("-")[1]) + 1, 1]
                rows.append(SimpleNamespace(index=index, embedding=vector))
            return SimpleNamespace(data=list(reversed(rows)))
        self.client.embeddings.create.side_effect = respond
        scores = self.score([f"doc-{i}" for i in range(60)])
        self.assertEqual(len(scores), 60)
        for i, score in enumerate(scores):
            self.assertAlmostEqual(score, (i + 1) / math.hypot(i + 1, 1))
        calls = self.client.embeddings.create.call_args_list
        self.assertEqual([len(call.kwargs["input"]) for call in calls], [10] * 6 + [1])
        self.assertTrue(all(call.kwargs["encoding_format"] == "float" for call in calls))
        self.assertEqual(self.factory.call_args.kwargs["max_retries"], 0)
        self.client.close.assert_called_once()
        self.local.assert_not_called()

    def test_incomplete_duplicate_and_invalid_vectors_reject_the_whole_result(self):
        invalid_rows = [
            [(0, [1, 0])],
            [(0, [1, 0]), (0, [1, 0])],
            [(0, [1, 0]), (2, [1, 0])],
            [(0, [1, 0]), (True, [1, 0])],
            [(0, [1, 0]), (1, [1])],
            [(0, [1, 0]), (1, [float("nan"), 0])],
            [(0, [1, 0]), (1, [0, 0])],
        ]
        for rows in invalid_rows:
            with self.subTest(rows=rows), self.assertLogs(level="WARNING"):
                self.client.embeddings.create.return_value = SimpleNamespace(data=[
                    SimpleNamespace(index=index, embedding=embedding) for index, embedding in rows
                ])
                self.assertIsNone(self.score(["doc"]))
        self.assertEqual(self.client.close.call_count, len(invalid_rows))

    def test_request_error_falls_back_without_logging_chat_or_credentials(self):
        self.client.embeddings.create.side_effect = RuntimeError("private chat test-only-key")
        with self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(self.score(["doc"]))
        self.assertNotIn("private chat", "".join(logs.output))
        self.assertNotIn("test-only-key", "".join(logs.output))
        self.client.close.assert_called_once()
        self.local.assert_not_called()

    def test_later_batch_failure_does_not_return_partial_scores(self):
        self.client.embeddings.create.side_effect = [
            SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1, 0]) for i in range(10)]),
            TimeoutError("request timed out"),
        ]
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(self.score(["doc"] * 10))
        self.client.close.assert_called_once()

    def test_budget_exhaustion_prevents_another_batch(self):
        self.client.embeddings.create.return_value = SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[1, 0]) for i in range(10)
        ])
        with patch("chat_reply_agent.services.embedding_service.monotonic", side_effect=[0, 0, 21]), \
                self.assertLogs(level="WARNING"):
            self.assertIsNone(self.score(["doc"] * 10))
        self.client.embeddings.create.assert_called_once()
        self.client.close.assert_called_once()

    def test_missing_credentials_or_foreign_endpoint_make_no_network_request(self):
        for overrides in ({"api_key": ""}, {"base_url": "https://example.com/v1"},
                          {"base_url": "http://dashscope.aliyuncs.com/v1"}):
            with self.subTest(overrides=overrides), self.assertLogs(level="WARNING"):
                self.assertIsNone(self.score(["doc"], **overrides))
        self.factory.assert_not_called()

    def test_no_history_skips_cloud_call(self):
        self.assertEqual(self.score([]), [])
        self.factory.assert_not_called()
