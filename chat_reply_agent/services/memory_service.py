import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_reply_agent.models import ConversationMemory
from chat_reply_agent.schemas import ChatAnalysis, ChatScenario


def get_latest_memory(
    db: Session,
    user_id: str,
    relation_type: str,
) -> ConversationMemory | None:
    statement = (
        select(ConversationMemory)
        .where(
            ConversationMemory.user_id == user_id,
            ConversationMemory.relation_type == relation_type,
        )
        .order_by(ConversationMemory.updated_at.desc())
        .limit(1)
    )
    return db.scalar(statement)


def get_memory_summary(db: Session, user_id: str, relation_type: str) -> str:
    memory = get_latest_memory(db, user_id, relation_type)
    return memory.summary if memory else ""


def save_conversation_memory(
    db: Session,
    user_id: str,
    relation_type: str,
    chat_text: str,
    analysis: ChatAnalysis,
    scenario: ChatScenario,
) -> ConversationMemory:
    summary = _build_summary(chat_text, analysis, scenario)
    memory = ConversationMemory(
        user_id=user_id,
        relation_type=relation_type,
        summary=summary,
        last_chat_text=chat_text,
        analysis_json=json.dumps(
            {
                "analysis": analysis.model_dump(),
                "scenario": scenario.model_dump(),
            },
            ensure_ascii=False,
        ),
    )
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory


def _build_summary(
    chat_text: str,
    analysis: ChatAnalysis,
    scenario: ChatScenario,
) -> str:
    clipped = chat_text.replace("\n", " ")[:180]
    points = "；".join(analysis.key_points[:3])
    return (
        f"场景：{scenario.label}。情绪：{analysis.emotion}。"
        f"态度：{analysis.attitude}。重点：{points or clipped}"
    )
