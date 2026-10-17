"""SQLite outbox for the single-process local app; never share request sessions."""

import json
import logging
import sqlite3
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from chat_reply_agent.database import SessionLocal, engine
from chat_reply_agent.models import MemoryJob
from chat_reply_agent.schemas import ChatAnalysis, ChatScenario, ExtractedChat
from chat_reply_agent.services.contact_memory_service import (
    apply_contact_memory, get_contact_snapshot, prepare_contact_memory,
)
from chat_reply_agent.services.llm_client import model_role
from chat_reply_agent.services.memory_service import save_conversation_memory
from chat_reply_agent.services.retrieval_service import save_memory_episode
from chat_reply_agent.services.style_service import learn_user_style
from chat_reply_agent.services.telemetry_service import log_trace, timed_call, trace_scope


logger = logging.getLogger("uvicorn.error")
_stop = Event()
_wake = Event()
_worker: Thread | None = None
_ownership: sqlite3.Connection | None = None


def acquire_worker_lock(database: str) -> sqlite3.Connection | None:
    connection = sqlite3.connect(f"{Path(database).resolve()}.worker-lock", timeout=0.1, check_same_thread=False)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS worker_lock (id INTEGER PRIMARY KEY)")
        connection.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError:
        connection.close()
        return None
    return connection


class AtomicMemorySession(Session):
    # Existing memory services commit independently. Flush them into one job transaction.
    def commit(self):
        self.flush()


def enqueue_memory(db: Session, *, user_id, contact_id, relation_type, **payload) -> str:
    job = MemoryJob(id=uuid4().hex, user_id=user_id, contact_id=contact_id,
                    relation_type=relation_type, payload_json=json.dumps(payload, ensure_ascii=False))
    db.add(job)
    db.commit()
    _wake.set()
    return job.id


def cancel_memory_jobs(db, *, user_id, contact_id=None, relation_type=None):
    conditions = [MemoryJob.user_id == user_id, MemoryJob.status.in_(["pending", "running", "failed"])]
    if contact_id is not None:
        conditions.append(MemoryJob.contact_id == contact_id)
    if relation_type is not None:
        conditions.append(MemoryJob.relation_type == relation_type)
    result = db.execute(update(MemoryJob).where(*conditions).values(
        status="cancelled", payload_json="{}", result_json="{}", error=""))
    return result.rowcount


def process_next_memory_job(session_factory=SessionLocal) -> bool:
    with session_factory() as db:
        job = db.scalar(select(MemoryJob).where(MemoryJob.status == "pending")
                        .order_by(MemoryJob.created_at, MemoryJob.id).limit(1))
        if job is None:
            return False
        job_id, user_id, contact_id, relation_type = job.id, job.user_id, job.contact_id, job.relation_type
        payload_json = job.payload_json
        claimed = db.execute(update(MemoryJob).where(
            MemoryJob.id == job_id, MemoryJob.status == "pending").values(status="running"))
        db.commit()
        if not claimed.rowcount:
            return True
        bind = db.get_bind()
    try:
        payload = json.loads(payload_json)
        analysis = ChatAnalysis.model_validate(payload["analysis"])
        scenario = ChatScenario.model_validate(payload["scenario"])
        chat = ExtractedChat.model_validate(payload["extracted_chat"])
        with session_factory() as db:
            previous = get_contact_snapshot(db, user_id, contact_id, payload["contact_name"], relation_type)
        with trace_scope() as trace:
            with model_role("memory"):
                values = timed_call("memory.contact", prepare_contact_memory, previous,
                    payload["contact_name"], relation_type, payload["chat_text"], chat.topic,
                    analysis, scenario, allow_fallback=False)
            # Acquire a write lock and re-check cancellation AFTER the network call.
            # This prevents deleted/edited memories being resurrected by a stale result.
            with AtomicMemorySession(bind=bind) as db:
                active = db.execute(update(MemoryJob).where(
                    MemoryJob.id == job_id, MemoryJob.status == "running").values(status="running"))
                if not active.rowcount:
                    db.rollback()
                    return True
                apply_contact_memory(db, user_id, contact_id, values)
                memory = timed_call("memory.summary", save_conversation_memory, db, user_id,
                                    relation_type, payload["chat_text"], analysis, scenario)
                timed_call("memory.episode", save_memory_episode, db=db, user_id=user_id,
                    contact_id=contact_id, relation_type=relation_type, topic=chat.topic or analysis.scene,
                    scenario=scenario, summary=memory.summary, chat_text=payload["chat_text"],
                    conversation_memory_id=memory.id)
                timed_call("style.learn", learn_user_style, db, user_id, relation_type, chat)
                result = {"saved_memory_id": memory.id, "diagnostics": trace.snapshot()}
                db.execute(update(MemoryJob).where(MemoryJob.id == job_id).values(
                    status="completed", payload_json="{}", result_json=json.dumps(result, ensure_ascii=False)))
                Session.commit(db)
            log_trace(trace, 200)
    except Exception as exc:
        with session_factory() as db:
            db.execute(update(MemoryJob).where(MemoryJob.id == job_id, MemoryJob.status == "running")
                       .values(status="failed", payload_json="{}", error=type(exc).__name__))
            db.commit()
        logger.warning("memory_job_failed job_id=%s error_type=%s", job_id, type(exc).__name__)
    return True


def start_memory_worker():
    global _worker, _ownership
    if _worker is not None and _worker.is_alive():
        return
    # A separate SQLite file owns the worker without holding a lock on chat data.
    # OS-released locks make crash recovery safe even during overlapping reloads.
    database = engine.url.database
    if database and database != ":memory:":
        _ownership = acquire_worker_lock(database)
        if _ownership is None:
            raise RuntimeError("Another memory worker owns this database; run one app process.")
    try:
        with SessionLocal() as db:
            db.execute(update(MemoryJob).where(MemoryJob.status == "running").values(status="pending"))
            db.commit()
        _stop.clear()
        _worker = Thread(target=_run, name="memory-worker", daemon=True)
        _worker.start()
    except Exception:
        if _ownership is not None:
            _ownership.close()
            _ownership = None
        _worker = None
        raise


def _run():
    while not _stop.is_set():
        try:
            if process_next_memory_job():
                continue
        except Exception as exc:
            logger.warning("memory_worker_error error_type=%s", type(exc).__name__)
        _wake.wait(1)
        _wake.clear()


def stop_memory_worker():
    global _worker, _ownership
    _stop.set()
    _wake.set()
    if _worker is not None:
        _worker.join()
        _worker = None
    if _ownership is not None:
        _ownership.close()
        _ownership = None
