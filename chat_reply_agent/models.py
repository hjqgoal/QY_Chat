from datetime import UTC, datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from chat_reply_agent.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class MemoryJob(Base):
    __tablename__ = "memory_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    contact_id: Mapped[str] = mapped_column(String(64), index=True)
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True, default="pending")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class ConversationMemory(Base):
    __tablename__ = "conversation_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    last_chat_text: Mapped[str] = mapped_column(Text, default="")
    analysis_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow,
        onupdate=_utcnow,
    )


class UserStyleProfile(Base):
    __tablename__ = "user_style_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    style_summary: Mapped[str] = mapped_column(Text, default="")
    example_phrases: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow,
        onupdate=_utcnow,
    )


class ChatCorrection(Base):
    __tablename__ = "chat_corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    corrected_chat_json: Mapped[str] = mapped_column(Text, default="{}")
    corrected_chat_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ReplyFeedback(Base):
    __tablename__ = "reply_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    feedback_type: Mapped[str] = mapped_column(String(32), index=True)
    candidate_index: Mapped[int] = mapped_column(Integer, default=0)
    candidate_text: Mapped[str] = mapped_column(Text, default="")
    edited_text: Mapped[str] = mapped_column(Text, default="")
    memory_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class StyleExample(Base):
    __tablename__ = "style_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    text: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32), default="accepted")
    feedback_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ContactMemory(Base):
    __tablename__ = "contact_memories"
    __table_args__ = (
        UniqueConstraint("user_id", "contact_id", name="uq_contact_memory_owner"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    contact_id: Mapped[str] = mapped_column(String(64), index=True)
    contact_name: Mapped[str] = mapped_column(String(64), default="")
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    facts_json: Mapped[str] = mapped_column(Text, default="[]")
    interaction_preferences_json: Mapped[str] = mapped_column(Text, default="[]")
    last_topic: Mapped[str] = mapped_column(Text, default="")
    last_scenario: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow,
        onupdate=_utcnow,
    )


class MemoryEpisode(Base):
    __tablename__ = "memory_episodes"
    __table_args__ = (
        Index(
            "ix_memory_episodes_owner_contact_created",
            "user_id",
            "contact_id",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    contact_id: Mapped[str] = mapped_column(String(64), index=True)
    relation_type: Mapped[str] = mapped_column(String(32), index=True)
    topic: Mapped[str] = mapped_column(Text, default="")
    scenario_type: Mapped[str] = mapped_column(String(64), default="")
    scenario_label: Mapped[str] = mapped_column(String(64), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    chat_text: Mapped[str] = mapped_column(Text, default="")
    keywords_json: Mapped[str] = mapped_column(Text, default="[]")
    conversation_memory_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
