from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_reply_agent.models import ReplyFeedback, StyleExample


POSITIVE_STYLE_SOURCES = {"accepted", "edited", "manual"}


def save_style_example_from_feedback(
    db: Session,
    feedback: ReplyFeedback,
) -> StyleExample | None:
    text = _style_text_from_feedback(feedback)
    if not text:
        return None

    existing = db.scalar(
        select(StyleExample).where(
            StyleExample.user_id == feedback.user_id,
            StyleExample.relation_type == feedback.relation_type,
            StyleExample.text == text,
        )
    )
    if existing:
        return existing

    example = StyleExample(
        user_id=feedback.user_id,
        relation_type=feedback.relation_type,
        text=text,
        source="accepted" if feedback.feedback_type == "accepted" else "edited",
        feedback_id=feedback.id,
    )
    db.add(example)
    db.commit()
    db.refresh(example)
    return example


def get_style_examples(
    db: Session,
    user_id: str,
    relation_type: str,
    limit: int = 8,
) -> list[StyleExample]:
    statement = (
        select(StyleExample)
        .where(
            StyleExample.user_id == user_id,
            StyleExample.relation_type == relation_type,
            StyleExample.source.in_(POSITIVE_STYLE_SOURCES),
        )
        .order_by(StyleExample.created_at.desc())
        .limit(max(1, min(limit, 50)))
    )
    return list(db.scalars(statement))


def get_style_examples_summary(
    db: Session,
    user_id: str,
    relation_type: str,
    limit: int = 8,
) -> str:
    examples = get_style_examples(db, user_id, relation_type, limit)
    if not examples:
        return ""

    sample_text = " / ".join(example.text for example in examples[:limit])
    return f"用户真实会发的表达样本：{sample_text}"


def create_manual_style_example(
    db: Session,
    user_id: str,
    relation_type: str,
    text: str,
) -> StyleExample:
    cleaned = _clean_text(text)
    existing = db.scalar(
        select(StyleExample).where(
            StyleExample.user_id == user_id,
            StyleExample.relation_type == relation_type,
            StyleExample.text == cleaned,
        )
    )
    if existing:
        return existing

    example = StyleExample(
        user_id=user_id,
        relation_type=relation_type,
        text=cleaned,
        source="manual",
    )
    db.add(example)
    db.commit()
    db.refresh(example)
    return example


def delete_style_example(
    db: Session,
    user_id: str,
    relation_type: str,
    example_id: int,
) -> bool:
    example = db.scalar(
        select(StyleExample).where(
            StyleExample.id == example_id,
            StyleExample.user_id == user_id,
            StyleExample.relation_type == relation_type,
        )
    )
    if example is None:
        return False
    feedback_id = example.feedback_id
    db.delete(example)
    if feedback_id is not None:
        feedback = db.scalar(
            select(ReplyFeedback).where(
                ReplyFeedback.id == feedback_id,
                ReplyFeedback.user_id == user_id,
                ReplyFeedback.relation_type == relation_type,
            )
        )
        if feedback is not None:
            db.delete(feedback)
    db.commit()
    return True


def _style_text_from_feedback(feedback: ReplyFeedback) -> str:
    edited = feedback.edited_text.strip()
    candidate = feedback.candidate_text.strip()

    if feedback.feedback_type == "accepted":
        return _clean_text(edited or candidate)

    if edited and edited != candidate:
        return _clean_text(edited)

    return ""


def _clean_text(text: str) -> str:
    return text.replace("\n", " ").strip()[:80]
