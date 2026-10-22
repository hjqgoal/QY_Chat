"""Request-local timing; no prompts, chat text, user IDs or credentials in logs."""

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from uuid import uuid4


logger = logging.getLogger("uvicorn.error")


@dataclass
class RequestTrace:
    request_id: str = field(default_factory=lambda: uuid4().hex)
    started: float = field(default_factory=perf_counter)
    stages: list[dict] = field(default_factory=list)
    model_calls: list[dict] = field(default_factory=list)
    validation_errors: list[dict] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "request_id": self.request_id,
            "total_ms": round(max(0, perf_counter() - self.started) * 1000, 2),
            "stages": list(self.stages),
            "model_calls": list(self.model_calls),
            "validation_errors": list(self.validation_errors),
        }


current_trace: ContextVar[RequestTrace | None] = ContextVar("request_trace", default=None)


@contextmanager
def trace_scope():
    trace = current_trace.get()
    token = None
    if trace is None:
        trace = RequestTrace()
        token = current_trace.set(trace)
    try:
        yield trace
    finally:
        if token is not None:
            current_trace.reset(token)


def timed_stage(name: str):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            trace = current_trace.get()
            if trace is None:
                return function(*args, **kwargs)
            started = perf_counter()
            status = "ok"
            try:
                result = function(*args, **kwargs)
                if name.startswith("model.") and result is None:
                    status = "skipped"
                return result
            except Exception:
                status = "error"
                raise
            finally:
                trace.stages.append({
                    "stage": name,
                    "duration_ms": round(max(0, perf_counter() - started) * 1000, 2),
                    "status": status,
                })
        return measured
    return decorate


def traced_response(function):
    @wraps(function)
    def measured(*args, **kwargs):
        with trace_scope() as trace:
            response = function(*args, **kwargs)
            response.diagnostics = trace.snapshot()
            return response
    return measured


def timed_call(name, function, *args, **kwargs):
    return timed_stage(name)(function)(*args, **kwargs)


def log_trace(trace: RequestTrace, status_code: int) -> None:
    logger.info("chat_timing %s", json.dumps({
        **trace.snapshot(), "http_status": status_code,
    }, ensure_ascii=True))
