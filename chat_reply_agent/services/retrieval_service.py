from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_reply_agent.config import Settings, get_settings
from chat_reply_agent.models import MemoryEpisode
from chat_reply_agent.schemas import ChatScenario, RetrievedMemoryItem, MemoryRetrievalDiagnostics
from chat_reply_agent.services.embedding_service import semantic_similarities, embedding_options
from chat_reply_agent.services.llm_client import ModelCallError


MAX_CANDIDATES = 60
DEFAULT_RETRIEVAL_LIMIT = 3
MIN_RELEVANCE_SCORE = 16
MIN_LEXICAL_RELEVANCE_SCORE = 10


def save_memory_episode(
    db: Session,
    *,
    user_id: str,
    contact_id: str,
    relation_type: str,
    topic: str,
    scenario: ChatScenario,
    summary: str,
    chat_text: str,
    conversation_memory_id: int | None = None,
) -> MemoryEpisode:
    episode_summary = _build_episode_summary(topic, summary)
    searchable_text = _join_non_empty(
        topic,
        scenario.label,
        episode_summary,
        chat_text,
    )
    episode = MemoryEpisode(
        user_id=user_id,
        contact_id=contact_id,
        relation_type=relation_type,
        topic=_clip(topic, 120),
        scenario_type=scenario.type,
        scenario_label=scenario.label,
        summary=episode_summary,
        chat_text=_clip(chat_text, 4000),
        keywords_json=json.dumps(
            _top_search_terms(searchable_text, limit=24),
            ensure_ascii=False,
        ),
        conversation_memory_id=conversation_memory_id,
    )
    db.add(episode)
    db.commit()
    db.refresh(episode)
    return episode


def retrieve_relevant_memories(
    db: Session,
    *,
    user_id: str,
    contact_id: str,
    query_text: str,
    relation_type: str = "",
    scenario_type: str = "",
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
    exclude_chat_text: str = "",
    diagnostics: MemoryRetrievalDiagnostics | None = None,
    settings: Settings | None = None,
) -> list[RetrievedMemoryItem]:
    diagnostics = diagnostics if diagnostics is not None else MemoryRetrievalDiagnostics()
    normalized_query = _normalize_text(query_text)
    if not normalized_query:
        diagnostics.message = "没有查询文本，未执行向量检索。"
        return []

    statement = (
        select(MemoryEpisode)
        .where(
            MemoryEpisode.user_id == user_id,
            MemoryEpisode.contact_id == contact_id,
        )
        .order_by(MemoryEpisode.created_at.desc(), MemoryEpisode.id.desc())
        .limit(MAX_CANDIDATES)
    )
    episodes = list(db.scalars(statement).all())
    normalized_excluded_chat = _conversation_fingerprint(exclude_chat_text)
    if normalized_excluded_chat:
        episodes = [
            episode
            for episode in episodes
            if _conversation_fingerprint(episode.chat_text)
            != normalized_excluded_chat
        ]
    if not episodes:
        diagnostics.message = "当前联系人没有可检索历史，未调用向量模型。"
        return []

    settings = settings if settings is not None else get_settings()
    diagnostics.candidate_count = len(episodes)
    similarities = None
    method = "lexical"
    if settings.memory_similarity_backend == "semantic":
        diagnostics.embedding_model = settings.memory_embedding_model
        strict = settings.memory_embedding_fallback == "error"
        try:
            similarities = semantic_similarities(
                query_text,
                [_join_non_empty(episode.topic, episode.summary) for episode in episodes],
                **embedding_options(settings), raise_on_error=strict,
            )
        except ModelCallError:
            if strict:
                raise
        if similarities is None and strict:
            raise ModelCallError("语义向量检索不可用，严格模式下不使用词面检索。", kind="embedding_unavailable")
        method = "semantic" if similarities is not None else "lexical_fallback"
    if similarities is None:
        similarities = _lexical_similarities(query_text, episodes)
    diagnostics.method = method
    diagnostics.message = {"semantic": "本次已完成语义向量检索。", "lexical": "本次使用词面检索。",
                           "lexical_fallback": "向量服务不可用，本次已降级为词面检索。"}[method]
    # Lexical scores no longer receive the old relation/scenario/recency bonuses.
    minimum = MIN_RELEVANCE_SCORE if method == "semantic" else MIN_LEXICAL_RELEVANCE_SCORE

    # One clock snapshot makes scores comparable, including at the threshold.
    now = datetime.now(UTC)
    scored: list[tuple[float, RetrievedMemoryItem]] = []
    for episode, similarity in zip(episodes, similarities, strict=True):
        half_life = settings.memory_decay_half_life_by_scenario.get(
            episode.scenario_type,
            settings.memory_decay_half_life_days,
        )
        decay = _time_decay_factor(episode.created_at, half_life, now=now)
        score = similarity * decay
        # Relation/scenario bonuses would violate the multiplicative formula.
        if score * 100 >= minimum:
            item = _to_item(episode, round(score * 100))
            item.similarity_score = similarity
            item.time_decay_factor = decay
            item.decay_half_life_days = half_life
            item.similarity_method = method
            scored.append((score, item))

    scored.sort(
        key=lambda entry: (entry[0], _as_utc(entry[1].created_at), entry[1].id),
        reverse=True,
    )
    results: list[RetrievedMemoryItem] = []
    seen_summaries: set[str] = set()
    for _, item in scored:
        summary_key = _normalize_text(item.summary)
        if summary_key and summary_key in seen_summaries:
            continue
        seen_summaries.add(summary_key)
        results.append(item)
        if len(results) >= max(1, min(limit, 10)):
            break
    diagnostics.retrieved_count = len(results)
    if not results:
        diagnostics.message += "没有达到相关性与时间衰减门槛的历史片段。"
    return results


def list_memory_episodes(
    db: Session,
    *,
    user_id: str,
    contact_id: str,
    limit: int = 10,
) -> list[RetrievedMemoryItem]:
    statement = (
        select(MemoryEpisode)
        .where(
            MemoryEpisode.user_id == user_id,
            MemoryEpisode.contact_id == contact_id,
        )
        .order_by(MemoryEpisode.created_at.desc(), MemoryEpisode.id.desc())
        .limit(max(1, min(limit, 30)))
    )
    return [_to_item(episode, 0) for episode in db.scalars(statement).all()]


def build_retrieval_context(
    contact_context: str,
    memories: list[RetrievedMemoryItem],
) -> str:
    parts: list[str] = []
    if contact_context:
        parts.append(f"联系人稳定记忆：\n{contact_context}")
    if memories:
        rows = []
        for item in memories:
            label = item.topic or item.scenario_label or "历史聊天"
            date = item.created_at.strftime("%Y-%m-%d")
            rows.append(
                f"- [{date}｜{label}｜相关度 {item.relevance_score}%] {item.summary}"
            )
        parts.append(
            "按当前话题召回的历史片段（只作事实参考，若与当前聊天冲突，"
            "以当前聊天为准）：\n" + "\n".join(rows)
        )
    return "\n\n".join(parts)


def _to_item(episode: MemoryEpisode, relevance: int) -> RetrievedMemoryItem:
    return RetrievedMemoryItem(
        id=episode.id,
        topic=episode.topic,
        scenario_type=episode.scenario_type,
        scenario_label=episode.scenario_label,
        summary=episode.summary,
        relevance_score=relevance,
        created_at=episode.created_at,
    )


def _episode_terms(episode: MemoryEpisode) -> Counter[str]:
    text = _join_non_empty(
        episode.topic,
        episode.scenario_label,
        episode.summary,
        episode.chat_text,
        " ".join(_load_keywords(episode.keywords_json)),
    )
    return Counter(_search_terms(text))


def _lexical_similarities(
    query_text: str,
    episodes: list[MemoryEpisode],
) -> list[float]:
    query_terms = _search_terms(query_text)
    documents = [_episode_terms(episode) for episode in episodes]
    frequencies = Counter(term for terms in documents for term in terms)
    return [
        _lexical_relevance(query_terms, terms, frequencies, len(documents))
        for terms in documents
    ]


def _lexical_relevance(
    query_terms: Counter[str],
    document_terms: Counter[str],
    document_frequency: Counter[str],
    document_count: int,
) -> float:
    if not query_terms or not document_terms:
        return 0.0
    shared = set(query_terms) & set(document_terms)
    if not shared:
        return 0.0

    def weight(term: str) -> float:
        # Unseen query fragments must not outweigh every term in the corpus.
        # Otherwise short follow-ups need an artificial repeated topic to match.
        return math.log((document_count + 1) / (max(1, document_frequency[term]) + 1)) + 1

    shared_weight = sum(
        min(query_terms[term], document_terms[term]) * weight(term)
        for term in shared
    )
    query_weight = sum(count * weight(term) for term, count in query_terms.items())
    document_weight = sum(
        count * weight(term) for term, count in document_terms.items()
    )
    coverage = shared_weight / query_weight if query_weight else 0.0
    precision = shared_weight / document_weight if document_weight else 0.0
    strongest_match = max(
        (
            min(query_terms[term], document_terms[term]) * weight(term)
            / query_weight
        )
        for term in shared
    )
    # A short Chinese topic can be represented by one repeated bigram, such as
    # "火锅". Preserve that strong signal without making generic one-off words
    # enough to pass the retrieval threshold on their own.
    return min(
        1.0,
        coverage * 0.78 + precision * 0.22 + strongest_match * 0.55,
    )


def _time_decay_factor(
    created_at: datetime,
    half_life_days: float,
    *,
    now: datetime | None = None,
) -> float:
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("Memory half-life must be finite and positive")
    age_days = max(
        0.0,
        (_as_utc(now or datetime.now(UTC)) - _as_utc(created_at)).total_seconds()
        / 86400,
    )
    return 2.0 ** (-age_days / half_life_days)


def _as_utc(value: datetime) -> datetime:
    # Existing SQLite timestamps are naive UTC, not the host's local timezone.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _search_terms(text: str) -> Counter[str]:
    normalized = (text or "").lower()
    terms: list[str] = []
    for latin in re.findall(r"[a-z0-9]{2,}", normalized):
        terms.append(latin)
    for chunk in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(chunk) == 1:
            terms.append(chunk)
            continue
        terms.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
        if len(chunk) >= 3:
            terms.extend(chunk[index : index + 3] for index in range(len(chunk) - 2))
    return Counter(terms)


def _top_search_terms(text: str, limit: int) -> list[str]:
    terms = _search_terms(text)
    return [term for term, _ in terms.most_common(limit)]


def _load_keywords(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalize_text(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", (text or "").lower()))


def _conversation_fingerprint(text: str) -> str:
    content: list[str] = []
    ignored_prefixes = ("对方最后一条消息", "识别备注")
    message_prefixes = ("我", "对方", "未知")
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(ignored_prefixes):
            continue
        if line.startswith("对话主题"):
            content.append(_after_first_colon(line))
            continue
        if line.startswith(message_prefixes) and (":" in line or "：" in line):
            content.append(_after_first_colon(line))
            continue
        content.append(line)
    return _normalize_text("\n".join(content))


def _after_first_colon(text: str) -> str:
    ascii_index = text.find(":")
    chinese_index = text.find("：")
    indexes = [index for index in (ascii_index, chinese_index) if index >= 0]
    if not indexes:
        return text
    return text[min(indexes) + 1 :].strip()


def _join_non_empty(*parts: str) -> str:
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def _clip(text: str, limit: int) -> str:
    return (text or "").strip()[:limit]


def _build_episode_summary(topic: str, summary: str) -> str:
    clean_topic = _clip(topic, 120)
    clean_summary = (summary or "").strip()
    if clean_topic and _normalize_text(clean_topic) not in _normalize_text(clean_summary):
        clean_summary = f"话题：{clean_topic}。{clean_summary}"
    return _clip(clean_summary or f"话题：{clean_topic or '未标注'}。", 320)
