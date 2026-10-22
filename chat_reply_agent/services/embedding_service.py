"""Local or API embeddings with explicit credentials and failure policy."""

import logging
import math
from importlib.util import find_spec
from functools import lru_cache
from time import monotonic
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from chat_reply_agent.services.telemetry_service import timed_stage
from chat_reply_agent.services.llm_client import ModelCallError


logger = logging.getLogger(__name__)


def _uses_dashscope_multimodal(provider: str, model_name: str) -> bool:
    return provider == "dashscope" and model_name == "qwen3-vl-embedding"


def embedding_runtime_status(settings) -> dict:
    policy = {"fallback_policy": settings.memory_embedding_fallback}
    if settings.memory_similarity_backend == "lexical":
        return {**policy, "readiness": "lexical", "notice": "使用词面检索，不是向量语义检索。"}
    try:
        embedding_options(settings)
    except ModelCallError as exc:
        return {**policy, "readiness": "configuration_missing", "notice": str(exc)}
    if settings.memory_embedding_provider == "local" and find_spec("sentence_transformers") is None:
        notice = "缺少本地向量依赖；" + ("需要检索历史时将报错。" if settings.memory_embedding_fallback == "error" else "检索将降级为词面匹配。")
        return {**policy, "readiness": "dependency_missing", "notice": notice}
    native = _uses_dashscope_multimodal(settings.memory_embedding_provider, settings.memory_embedding_model.strip())
    label = "语义检索（百炼原生多模态向量接口）" if native else "语义检索"
    return {**policy, "readiness": "not_verified", "notice": f"已配置{label}，尚未验证连通性；实际方式以本次检索结果为准。"}


def _validate_api_url(base_url: str, provider: str) -> None:
    try:
        endpoint = urlsplit(base_url)
        endpoint.port  # Validate malformed ports without exposing the URL.
    except ValueError as exc:
        raise ModelCallError("MEMORY_EMBEDDING_BASE_URL 格式无效，请检查地址。", kind="embedding_configuration") from exc
    if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment or "{" in base_url or "}" in base_url
            or endpoint.path.rstrip("/").endswith(("/embeddings", "/chat/completions"))):
        raise ModelCallError("请配置有效的 MEMORY_EMBEDDING_BASE_URL（HTTPS 基础地址，不含 /embeddings）。", kind="embedding_configuration")
    if provider == "dashscope" and not endpoint.hostname.endswith(".aliyuncs.com"):
        raise ModelCallError("dashscope 模式需要百炼地址；其他兼容服务请选择 openai_compatible 并单独填写密钥。", kind="embedding_configuration")


def _dashscope_multimodal_url(base_url: str) -> str:
    _validate_api_url(base_url, "dashscope")
    endpoint = urlsplit(base_url)
    if endpoint.path.rstrip("/") not in {"/compatible-mode/v1", "/api/v1"}:
        raise ModelCallError(
            "qwen3-vl-embedding 需要百炼基础地址，以 /compatible-mode/v1 或 /api/v1 结尾；不要填写完整模型接口。",
            kind="embedding_configuration",
        )
    # Switch only the API path; never move credentials to another host or region.
    return endpoint._replace(path="/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding").geturl()


def embedding_options(settings) -> dict:
    options = dict(model_name=settings.memory_embedding_model.strip(), provider=settings.memory_embedding_provider,
                   timeout=settings.memory_embedding_timeout_seconds)
    if not options["model_name"]:
        raise ModelCallError("请填写 MEMORY_EMBEDDING_MODEL，不能使用聊天模型代替向量模型。", kind="embedding_configuration")
    if settings.memory_embedding_provider == "local":
        return {**options, "api_key": None, "base_url": ""}
    reuse = settings.memory_embedding_reuse_chat_credentials
    base_url = (settings.memory_embedding_base_url or (settings.openai_base_url if reuse else "")).strip().rstrip("/")
    _validate_api_url(base_url, settings.memory_embedding_provider)
    if _uses_dashscope_multimodal(options["provider"], options["model_name"]):
        _dashscope_multimodal_url(base_url)
    api_key = (settings.memory_embedding_api_key or "").strip()
    if not api_key and reuse:
        if base_url != settings.openai_base_url.strip().rstrip("/"):
            raise ModelCallError("向量地址与聊天地址不同，不能复用聊天密钥，请单独填写 MEMORY_EMBEDDING_API_KEY。", kind="embedding_configuration")
        api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        raise ModelCallError("请在 .env 填写 MEMORY_EMBEDDING_API_KEY；当前没有可用的向量密钥。", kind="embedding_configuration")
    return {**options, "api_key": api_key, "base_url": base_url}


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
            "Local memory embeddings unavailable (%s). "
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
    raise_on_error: bool = False,
) -> list[float] | None:
    if not documents:
        return []
    try:
        if provider in {"dashscope", "openai_compatible"}:
            vectors = _api_vectors(
                [query, *documents], model_name=model_name,
                api_key=api_key, base_url=base_url, timeout=timeout, provider=provider,
            )
        elif provider == "local":
            model = _load_model(model_name)
            if model is None:
                raise ModelCallError("本地向量模型不可用，请配置 Embedding API 或安装依赖并准备本地模型。", kind="embedding_unavailable")
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
        if raise_on_error:
            if isinstance(exc, ModelCallError):
                raise
            raise ModelCallError(
                f"语义向量检索失败（{type(exc).__name__}），请检查向量服务的地址、密钥、模型和额度。未退回词面检索。",
                kind="embedding_unavailable",
            ) from exc
        logger.warning(
            "Memory embedding failed (%s); using lexical similarity.",
            type(exc).__name__,
        )
        return None


def _api_vectors(texts, *, model_name, api_key, base_url, timeout, provider):
    _validate_api_url(base_url, provider)
    if not api_key or not api_key.strip():
        raise ValueError("Embedding API key is not configured")
    if _uses_dashscope_multimodal(provider, model_name):
        return _dashscope_multimodal_vectors(
            texts, model_name=model_name, api_key=api_key, base_url=base_url, timeout=timeout,
        )
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


def _dashscope_multimodal_vectors(texts, *, model_name, api_key, base_url, timeout):
    url = _dashscope_multimodal_url(base_url)
    deadline = monotonic() + timeout
    vectors = []
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        # One independent vector per text, including the query; fusion would break retrieval.
        for offset in range(0, len(texts), 20):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Embedding request budget exhausted")
            batch = texts[offset:offset + 20]
            response = client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name,
                    "input": {"contents": [{"text": text} for text in batch]},
                    "parameters": {"enable_fusion": False, "output_type": "dense"},
                },
                timeout=remaining,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                hints = {
                    400: "请检查模型参数及该地域是否支持此模型。",
                    401: "请检查 MEMORY_EMBEDDING_API_KEY 是否有效且属于该地域。",
                    403: "请检查向量密钥的模型权限及业务空间。",
                    404: "请检查百炼基础地址、地域及 qwen3-vl-embedding 的可用性。",
                    429: "请检查向量服务的限流和可用额度，稍后重试。",
                }
                hint = hints.get(response.status_code, "请检查向量服务状态，稍后重试。")
                # Provider bodies may echo credentials or private text; never surface them.
                raise ModelCallError(
                    f"百炼原生向量接口请求失败（HTTP {response.status_code}）。{hint}",
                    kind="embedding_unavailable",
                ) from exc
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("code"):
                raise ModelCallError("百炼向量服务返回异常结果，未生成可用向量。", kind="embedding_unavailable")
            output = payload.get("output")
            rows = output.get("embeddings") if isinstance(output, dict) else None
            if (not isinstance(rows, list) or len(rows) != len(batch)
                    or any(not isinstance(row, dict) or type(row.get("index")) is not int
                           or not isinstance(row.get("embedding"), list)
                           or row.get("type") not in {"text", "vl"} for row in rows)
                    or {row["index"] for row in rows} != set(range(len(batch)))):
                raise ModelCallError(
                    "百炼向量结果的条数、下标或类型异常；需要每段文字的独立向量，不能使用融合结果。",
                    kind="embedding_unavailable",
                )
            vectors.extend(row["embedding"] for row in sorted(rows, key=lambda row: row["index"]))
    return vectors
