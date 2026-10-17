from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from chat_reply_agent.models import ReplyFeedback
from chat_reply_agent.schemas import ReplyFeedbackRequest


POSITIVE_FEEDBACK = {"accepted"}

FEEDBACK_LABELS = {
    "accepted": "采纳",
    "too_ai": "太 AI",
    "too_long": "太长",
    "too_oily": "太油",
    "too_cold": "太冷淡",
    "irrelevant": "答非所问",
    "wrong_relation": "不符合关系",
}


def save_reply_feedback(
    db: Session,
    payload: ReplyFeedbackRequest,
) -> ReplyFeedback:
    feedback = ReplyFeedback(
        user_id=payload.user_id,
        relation_type=payload.relation_type,
        feedback_type=payload.feedback_type,
        candidate_index=payload.candidate_index,
        candidate_text=payload.candidate_text,
        edited_text=payload.edited_text or "",
        memory_id=payload.memory_id,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_reply_preference_summary(
    db: Session,
    user_id: str,
    relation_type: str,
    limit: int = 30,
) -> str:
    rows = _recent_feedback(db, user_id, relation_type, limit)
    if not rows:
        return ""

    accepted_texts: list[str] = []
    disliked = Counter()
    edited_texts: list[str] = []

    for row in rows:
        if row.feedback_type in POSITIVE_FEEDBACK:
            accepted_texts.append(row.edited_text or row.candidate_text)
        else:
            disliked[row.feedback_type] += 1

        if row.edited_text and row.edited_text != row.candidate_text:
            edited_texts.append(row.edited_text)

    parts: list[str] = []
    if accepted_texts:
        examples = " / ".join(_clip_text(item) for item in accepted_texts[:4])
        parts.append(f"用户采纳过这些表达：{examples}")

    if edited_texts:
        examples = " / ".join(_clip_text(item) for item in edited_texts[:4])
        parts.append(f"用户手动改成过：{examples}")

    if disliked:
        labels = [
            f"{FEEDBACK_LABELS.get(key, key)} {count} 次"
            for key, count in disliked.most_common(4)
        ]
        parts.append(f"用户不喜欢：{'、'.join(labels)}")

    if not parts:
        return ""

    return "；".join(parts)


def _recent_feedback(
    db: Session,
    user_id: str,
    relation_type: str,
    limit: int,
) -> list[ReplyFeedback]:
    statement = (
        select(ReplyFeedback)
        .where(
            ReplyFeedback.user_id == user_id,
            ReplyFeedback.relation_type == relation_type,
        )
        .order_by(ReplyFeedback.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(statement))


def _clip_text(text: str, limit: int = 28) -> str:
    cleaned = text.replace("\n", " ").strip()
    return cleaned if len(cleaned) <= limit else f"{cleaned[:limit]}..."
