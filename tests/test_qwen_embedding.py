from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import httpx

from chat_reply_agent.config import Settings
from chat_reply_agent.services.embedding_service import (
    _dashscope_multimodal_url, embedding_options, embedding_runtime_status, semantic_similarities,
)
from chat_reply_agent.services.llm_client import ModelCallError
from scripts.check_embedding_connection import QUERY, RELEVANT, run_probe


HTTP_CLIENT = httpx.Client
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
NATIVE_PATH = "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"


def settings(**overrides):
    values = dict(_env_file=None, memory_similarity_backend="semantic", memory_embedding_provider="dashscope",
                  memory_embedding_model="qwen3-vl-embedding", memory_embedding_api_key="synthetic-test-secret",
                  memory_embedding_base_url=BASE_URL, memory_embedding_fallback="error")
    values.update(overrides)
    return Settings(**values)


def payload(vectors):
    return {"output": {"embeddings": [
        {"index": index, "embedding": vector, "type": "vl"}
        for index, vector in reversed(list(enumerate(vectors)))
    ]}}


@contextmanager
def native_api(handler):
    clients = []

    def factory(**kwargs):
        client = HTTP_CLIENT(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    with patch("chat_reply_agent.services.embedding_service.httpx.Client", side_effect=factory):
        yield clients


class QwenEmbeddingTests(TestCase):
    def similarities(self, query="query", documents=None, config=None):
        return semantic_similarities(query, documents if documents is not None else ["document"],
                                     **embedding_options(config or settings()), raise_on_error=True)

    def test_native_request_body_auth_order_and_cosine(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=payload([[2, 0], [4, 0], [0, 3]]))

        with native_api(handler) as clients, patch("chat_reply_agent.services.embedding_service.OpenAI") as compatible:
            scores = self.similarities(documents=["similar", "unrelated"])
        compatible.assert_not_called()
        self.assertEqual(scores, [1.0, 0.0])
        self.assertTrue(clients[0].is_closed)
        request = requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), "https://dashscope.aliyuncs.com" + NATIVE_PATH)
        self.assertEqual(request.headers["authorization"], "Bearer synthetic-test-secret")
        self.assertEqual(json.loads(request.content), {
            "model": "qwen3-vl-embedding",
            "input": {"contents": [{"text": "query"}, {"text": "similar"}, {"text": "unrelated"}]},
            "parameters": {"enable_fusion": False, "output_type": "dense"},
        })

    def test_native_batches_twenty_texts_and_preserves_cross_batch_order(self):
        sizes = []

        def handler(request):
            contents = json.loads(request.content)["input"]["contents"]
            sizes.append(len(contents))
            vectors = [[1, 0] if item["text"] == "query" or int(item["text"]) % 2 == 0 else [0, 1] for item in contents]
            return httpx.Response(200, json=payload(vectors))

        with native_api(handler):
            scores = self.similarities(documents=[str(i) for i in range(60)])
        self.assertEqual(sizes, [20, 20, 20, 1])
        self.assertEqual(scores, [1.0 if i % 2 == 0 else 0.0 for i in range(60)])

    def test_both_base_formats_keep_region_workspace_and_host(self):
        for host in ("dashscope.aliyuncs.com", "test-workspace.cn-beijing.maas.aliyuncs.com"):
            for path in ("/compatible-mode/v1", "/api/v1/"):
                base = f"https://{host}{path}"
                with self.subTest(base=base):
                    self.assertEqual(_dashscope_multimodal_url(base), f"https://{host}{NATIVE_PATH}")
                    self.assertEqual(embedding_options(settings(memory_embedding_base_url=base))["base_url"], base.rstrip("/"))

    def test_invalid_base_is_rejected_before_network(self):
        for url in ("https://example.test/api/v1", "http://dashscope.aliyuncs.com/api/v1",
                    "https://dashscope.aliyuncs.com/wrong", "https://dashscope.aliyuncs.com" + NATIVE_PATH,
                    "https://dashscope.aliyuncs.com/api/v1?token=secret"):
            with self.subTest(url=url), patch("chat_reply_agent.services.embedding_service.httpx.Client") as factory:
                with self.assertRaises(ModelCallError):
                    embedding_options(settings(memory_embedding_base_url=url))
                factory.assert_not_called()

    def test_generic_compatible_service_does_not_get_dashscope_native_payload(self):
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(data=[
            SimpleNamespace(index=0, embedding=[1, 0]), SimpleNamespace(index=1, embedding=[1, 0]),
        ])
        config = settings(memory_embedding_provider="openai_compatible", memory_embedding_base_url="https://example.test/v1")
        with patch("chat_reply_agent.services.embedding_service.OpenAI", return_value=client), \
                patch("chat_reply_agent.services.embedding_service.httpx.Client") as native:
            self.assertEqual(self.similarities(config=config), [1.0])
        native.assert_not_called()
        self.assertEqual(client.embeddings.create.call_args.kwargs["model"], "qwen3-vl-embedding")
        client.close.assert_called_once()

    def test_malformed_or_fused_results_are_not_used(self):
        valid = payload([[1, 0], [1, 0]])
        cases = [None, [], {}, {"output": None}, {"output": {"embeddings": []}}]
        for field, value in (("index", 0), ("index", True), ("index", 5),
                             ("type", "fusion"), ("embedding", None), ("embedding", []),
                             ("embedding", [0, 0]), ("embedding", [1, 0, 0]),
                             ("embedding", [float("nan"), 0]), ("embedding", [float("inf"), 0])):
            changed = json.loads(json.dumps(valid))
            changed["output"]["embeddings"][0][field] = value
            cases.append(changed)
        for value in cases:
            with self.subTest(value=value), native_api(lambda request: httpx.Response(200, content=json.dumps(value).encode())) as clients:
                with self.assertRaises(ModelCallError):
                    self.similarities()
            self.assertTrue(clients[0].is_closed)

    def test_http_failures_are_safe_and_do_not_follow_redirects(self):
        for status in (302, 400, 401, 403, 404, 429, 500):
            requests = []

            def handler(request):
                requests.append(request)
                return httpx.Response(status, headers={"location": "https://unrelated.example.test/"},
                                      json={"message": "synthetic-test-secret private-chat-text"})

            with self.subTest(status=status), native_api(handler) as clients, self.assertRaises(ModelCallError) as error:
                self.similarities()
            self.assertIn(f"HTTP {status}", str(error.exception))
            self.assertNotIn("synthetic-test-secret", str(error.exception))
            self.assertNotIn("private-chat-text", str(error.exception))
            self.assertEqual(len(requests), 1)
            self.assertTrue(clients[0].is_closed)

    def test_error_envelope_and_non_json_response_fail_safely(self):
        for body in (json.dumps({"code": "InvalidApiKey", "message": "synthetic-test-secret private-chat-text"}),
                     "<html>synthetic-test-secret private-chat-text</html>"):
            with self.subTest(body=body), native_api(lambda request: httpx.Response(200, content=body)), \
                    self.assertRaises(ModelCallError) as error:
                self.similarities()
            self.assertNotIn("synthetic-test-secret", str(error.exception))
            self.assertNotIn("private-chat-text", str(error.exception))

    def test_shared_timeout_exhaustion_does_not_return_partial_vectors(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=payload([[1, 0]] * 20))

        with native_api(handler) as clients, patch("chat_reply_agent.services.embedding_service.monotonic", side_effect=[0, 1, 21]):
            with self.assertRaises(ModelCallError):
                self.similarities(documents=["document"] * 20)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].extensions["timeout"]["read"], 19)
        self.assertTrue(clients[0].is_closed)

    def test_transport_timeout_is_safe_and_explicit_fallback_still_works(self):
        def handler(request):
            raise httpx.ReadTimeout("synthetic-test-secret private-chat-text", request=request)

        with native_api(handler), self.assertRaises(ModelCallError) as error:
            self.similarities()
        self.assertNotIn("private-chat-text", str(error.exception))
        with native_api(handler):
            self.assertIsNone(semantic_similarities("query", ["document"], **embedding_options(settings())))

    def test_empty_documents_and_configuration_status_make_no_request(self):
        with patch("chat_reply_agent.services.embedding_service.httpx.Client") as factory:
            status = embedding_runtime_status(settings())
            self.assertEqual(status["readiness"], "not_verified")
            self.assertEqual(self.similarities(documents=[]), [])
        factory.assert_not_called()

    def test_native_embeddings_drive_synthetic_rag_with_contact_isolation(self):
        inputs = []

        def handler(request):
            texts = [item["text"] for item in json.loads(request.content)["input"]["contents"]]
            inputs.extend(texts)
            vectors = [[1, 0] if text == QUERY or RELEVANT in text else [0, 1] for text in texts]
            return httpx.Response(200, json=payload(vectors))

        with native_api(handler):
            result = run_probe(settings())
        self.assertTrue(result["semantic_rag_verified"])
        self.assertTrue(result["contact_isolation_passed"])
        self.assertTrue(result["memory_added_to_generation_context"])
        self.assertEqual(result["retrieval"]["method"], "semantic")
        self.assertEqual(len(inputs), 3)
