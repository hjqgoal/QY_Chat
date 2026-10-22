"""Check semantic RAG with synthetic memories; never reads the personal DB."""

import argparse
import json
from pathlib import Path
import sys
from time import monotonic


QUERY = "晚饭想吃点不刺激的，有什么建议？"
RELEVANT = "对方吃辣容易胃不舒服，平时偏好清淡的饭菜。"


def run_probe(settings):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from chat_reply_agent.database import Base
    from chat_reply_agent.schemas import ChatScenario, MemoryRetrievalDiagnostics
    from chat_reply_agent.services.llm_client import ModelCallError
    from chat_reply_agent.services.retrieval_service import (
        save_memory_episode, retrieve_relevant_memories, build_retrieval_context,
    )

    if settings.memory_similarity_backend != "semantic":
        raise ModelCallError("请设置 MEMORY_SIMILARITY_BACKEND=semantic 后再检测。", kind="embedding_configuration")
    # The probe must fail on embedding errors even if the app permits fallback.
    strict_settings = settings.model_copy(update={"memory_embedding_fallback": "error"})
    engine = create_engine("sqlite://")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as db:
            def save(contact, topic, summary):
                return save_memory_episode(db, user_id="synthetic-probe", contact_id=contact,
                    relation_type="friend", topic=topic, summary=summary, chat_text=summary,
                    scenario=ChatScenario(type="daily_chat", label="合成测试", confidence=1))
            relevant = save("demo", "饮食偏好", RELEVANT)
            save("demo", "游戏计划", "对方计划周末与朋友组队打游戏。")
            save("another-contact", "隔离对照", "另一位联系人的记录，不得加入本次检索。")
            diagnostics = MemoryRetrievalDiagnostics()
            started = monotonic()
            hits = retrieve_relevant_memories(db, user_id="synthetic-probe", contact_id="demo",
                query_text=QUERY, diagnostics=diagnostics, settings=strict_settings)
            context = build_retrieval_context("", hits)
            semantic = diagnostics.method == "semantic"
            ranked_first = bool(hits) and hits[0].id == relevant.id
            injected = RELEVANT in context
            isolated = "另一位联系人" not in context and diagnostics.candidate_count == 2
            return {
                "provider": settings.memory_embedding_provider, "model": settings.memory_embedding_model,
                "synthetic_only": True, "personal_database_used": False,
                "elapsed_seconds": round(monotonic() - started, 3),
                "retrieval": diagnostics.model_dump(),
                "relevant_memory_ranked_first": ranked_first,
                "memory_added_to_generation_context": injected, "contact_isolation_passed": isolated,
                "semantic_rag_verified": semantic and ranked_first and injected and isolated,
                "note": "单个合成用例验证，不代表所有场景的召回准确率；未调用回复生成模型。",
            }
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="检查 Embedding 配置；--live 才会发送合成文本调用向量模型。")
    parser.add_argument("--live", action="store_true", help="允许向量请求，API 可能产生费用；不调用聊天模型")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from chat_reply_agent.config import get_settings
    from chat_reply_agent.services.embedding_service import embedding_runtime_status
    from chat_reply_agent.services.llm_client import ModelCallError

    try:
        settings = get_settings()
        if not args.live:
            status = embedding_runtime_status(settings)
            print(json.dumps({"live_called": False, **status}, ensure_ascii=False, indent=2))
            return 0 if status["readiness"] == "not_verified" else 1
        result = run_probe(settings)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["semantic_rag_verified"] else 1
    except ModelCallError as exc:
        print(json.dumps({"semantic_rag_verified": False, "error": str(exc), "error_kind": exc.kind}, ensure_ascii=False))
        return 1
    except Exception as exc:
        # Never echo validation inputs, provider exception bodies or secrets.
        print(json.dumps({"semantic_rag_verified": False, "error_kind": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
