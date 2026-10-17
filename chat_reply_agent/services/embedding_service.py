"""Local or DashScope embeddings, with explicit lexical fallback on failure."""

import logging
import math
from functools import lru_cache
from time import monotonic
from urllib.parse import urlsplit

from openai import OpenAI

from chat_reply_agent.services.telemetry_service import timed_stage


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            model_name,
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        logger.warning(
            "Local memory embeddings unavailable (%s); using lexical similarity. "
            "Install the optional dependency and cache the model, then restart.",
            type(exc).__name__,
        )
        return None


@timed_stage("memory.embedding")
def semantic_similarities(
    query: str,
    documents: list[str],
    *,
    model_name: str,
    provider: str = "local",
    api_key: str | None = None,
    base_url: str = "",
    timeout: float = 20,
) -> list[float] | None:
    if not documents:
        return []
    try:
        if provider == "dashscope":
            vectors = _dashscope_vectors(
                [query, *documents], model_name=model_name,
                api_key=api_key, base_url=base_url, timeout=timeout,
            )
        elif provider == "local":
            model = _load_model(model_name)
            if model is None:
                return None
            # Only the model is cached, never private chat text.
            vectors = model.encode(
                [query, *documents], batch_size=32,
                normalize_embeddings=True, show_progress_bar=False,
            )
        else:
            raise ValueError("Unknown embedding provider")
        if len(vectors) != len(documents) + 1:
            raise ValueError("Unexpected embedding count")
        # API embeddings need not be normalized. Normalize both providers before cosine.
        normalized = []
        dimensions = len(vectors[0])
        for vector in vectors:
            values = [float(value) for value in vector]
            if not dimensions or len(values) != dimensions or not all(map(math.isfinite, values)):
                raise ValueError("Invalid embedding dimensions or values")
            norm = math.hypot(*values)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Invalid embedding norm")
            normalized.append([value / norm for value in values])
        query_vector = normalized[0]
        scores = []
        for vector in normalized[1:]:
            score = sum(
                float(left) * float(right)
                for left, right in zip(query_vector, vector, strict=True)
            )
            if not math.isfinite(score):
                raise ValueError("Non-finite similarity")
            scores.append(max(0.0, min(1.0, score)))
        return scores
    except Exception as exc:
        logger.warning(
            "Memory embedding failed (%s); using lexical similarity.",
            type(exc).__name__,
        )
        return None


def _dashscope_vectors(texts, *, model_name, api_key, base_url, timeout):
    # Prevent accidentally sending the inherited chat credential to another provider.
    endpoint = urlsplit(base_url)
    if (endpoint.scheme != "https" or not endpoint.hostname
            or not endpoint.hostname.endswith(".aliyuncs.com")
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("DashScope embeddings require an Alibaba Cloud HTTPS endpoint")
    if not api_key or not api_key.strip():
        raise ValueError("Embedding API key is not configured")
    deadline = monotonic() + timeout
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    vectors = []
    try:
        # text-embedding-v4 accepts at most 10 texts per request, including the query.
        for offset in range(0, len(texts), 10):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Embedding request budget exhausted")
            batch = texts[offset:offset + 10]
            response = client.embeddings.create(
                model=model_name, input=batch, encoding_format="float", timeout=remaining,
            )
            rows = response.data
            if (len(rows) != len(batch)
                    or any(type(row.index) is not int for row in rows)
                    or {row.index for row in rows} != set(range(len(batch)))):
                raise ValueError("Missing or duplicate embedding indexes")
            vectors.extend(row.embedding for row in sorted(rows, key=lambda row: row.index))
    finally:
        client.close()
    return vectors
