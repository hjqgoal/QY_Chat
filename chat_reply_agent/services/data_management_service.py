from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from chat_reply_agent.models import (
    ContactMemory,
    ConversationMemory,
    MemoryEpisode,
    ReplyFeedback,
    StyleExample,
    UserStyleProfile,
    MemoryJob,
)
from chat_reply_agent.schemas import ContactMemoryDeleteResponse, StyleResetResponse


def delete_memory_episode_data(
    db: Session,
    *,
    user_id: str,
    contact_id: str,
    memory_id: int,
) -> bool:
    episode = db.scalar(
        select(MemoryEpisode).where(
            MemoryEpisode.id == memory_id,
            MemoryEpisode.user_id == user_id,
            MemoryEpisode.contact_id == contact_id,
        )
    )
    if episode is None:
        return False

    conversation_memory_id = episode.conversation_memory_id
    db.delete(episode)
    db.flush()
    if conversation_memory_id is not None:
        remaining_reference = db.scalar(
            select(MemoryEpisode.id)
            .where(MemoryEpisode.conversation_memory_id == conversation_memory_id)
            .limit(1)
        )
        if remaining_reference is None:
            db.execute(
                delete(ConversationMemory).where(
                    ConversationMemory.id == conversation_memory_id,
                    ConversationMemory.user_id == user_id,
                )
            )
    db.commit()
    return True


def delete_contact_memory_data(
    db: Session,
    *,
    user_id: str,
    contact_id: str,
) -> ContactMemoryDeleteResponse:
    from chat_reply_agent.services.memory_queue_service import cancel_memory_jobs
    cancelled = cancel_memory_jobs(db, user_id=user_id, contact_id=contact_id)
    db.execute(delete(MemoryJob).where(MemoryJob.user_id == user_id, MemoryJob.contact_id == contact_id))
    episodes = list(
        db.scalars(
            select(MemoryEpisode).where(
                MemoryEpisode.user_id == user_id,
                MemoryEpisode.contact_id == contact_id,
            )
        ).all()
    )
    conversation_ids = {
        episode.conversation_memory_id
        for episode in episodes
        if episode.conversation_memory_id is not None
    }
    contact_memory_count = len(
        list(
            db.scalars(
                select(ContactMemory.id).where(
                    ContactMemory.user_id == user_id,
                    ContactMemory.contact_id == contact_id,
                )
            ).all()
        )
    )

    if episodes:
        db.execute(
            delete(MemoryEpisode).where(
                MemoryEpisode.user_id == user_id,
                MemoryEpisode.contact_id == contact_id,
            )
        )

    deleted_conversation_memories = 0
    for conversation_id in conversation_ids:
        remaining_reference = db.scalar(
            select(MemoryEpisode.id)
            .where(MemoryEpisode.conversation_memory_id == conversation_id)
            .limit(1)
        )
        if remaining_reference is not None:
            continue
        result = db.execute(
            delete(ConversationMemory).where(
                ConversationMemory.id == conversation_id,
                ConversationMemory.user_id == user_id,
            )
        )
        deleted_conversation_memories += max(0, result.rowcount or 0)

    if contact_memory_count:
        db.execute(
            delete(ContactMemory).where(
                ContactMemory.user_id == user_id,
                ContactMemory.contact_id == contact_id,
            )
        )
    db.commit()
    return ContactMemoryDeleteResponse(
        contact_id=contact_id,
        deleted_contact_memory=contact_memory_count,
        deleted_memory_episodes=len(episodes),
        deleted_conversation_memories=deleted_conversation_memories,
        cancelled_memory_jobs=cancelled,
    )


def reset_style_data(
    db: Session,
    *,
    user_id: str,
    relation_type: str,
) -> StyleResetResponse:
    from chat_reply_agent.services.memory_queue_service import cancel_memory_jobs
    cancel_memory_jobs(db, user_id=user_id, relation_type=relation_type)
    filters = {
        UserStyleProfile: (
            UserStyleProfile.user_id == user_id,
            UserStyleProfile.relation_type == relation_type,
        ),
        StyleExample: (
            StyleExample.user_id == user_id,
            StyleExample.relation_type == relation_type,
        ),
        ReplyFeedback: (
            ReplyFeedback.user_id == user_id,
            ReplyFeedback.relation_type == relation_type,
        ),
    }
    counts: dict[type, int] = {}
    for model, conditions in filters.items():
        counts[model] = len(
            list(db.scalars(select(model.id).where(*conditions)).all())
        )
        if counts[model]:
            db.execute(delete(model).where(*conditions))
    db.commit()
    return StyleResetResponse(
        relation_type=relation_type,
        deleted_profiles=counts[UserStyleProfile],
        deleted_examples=counts[StyleExample],
        deleted_feedback=counts[ReplyFeedback],
    )
