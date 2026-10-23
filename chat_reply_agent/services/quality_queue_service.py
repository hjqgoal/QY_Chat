"""Bounded, ephemeral review -> optional repair -> recheck jobs."""

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import logging
from threading import Event, RLock, Thread
from time import monotonic
from uuid import uuid4

from chat_reply_agent.config import get_settings
from chat_reply_agent.schemas import QualityJobResponse, ReplyCandidate, ReplyControls, ReplyQualityReport
from chat_reply_agent.services.llm_client import ModelCallError, model_request_timeout, quality_model_limits
from chat_reply_agent.services.quality_service import review_candidates
from chat_reply_agent.services.quality_repair_service import repair_candidates, repair_targets, requires_quality_gate
from chat_reply_agent.services.telemetry_service import current_trace, trace_scope, timed_call


logger = logging.getLogger("uvicorn.error")
ACTIVE = {"pending", "running"}
MESSAGES = {
    "pending": "评估排队中，当前回复尚未验证。",
    "running": "正在评估原文，不会修改输入框。",
    "completed": "原文评估完成，仅供参考。",
    "failed": "评估未完成，保留原稿；请查看失败原因。",
    "timed_out": "评估超时，本任务已结束，不会自动重试；原稿未通过验证。",
    "cancelled": "评估已取消，旧结果不会再应用。",
    "busy": "评估队列已满，本轮未排队；稍后可重试。",
    "unavailable": "评估暂不可用，保留原稿；可重新评估。",
    "disabled": "备用示例不进行模型质量评估。",
}
ERRORS = {
    "queue_timeout": "等待空闲评估线程超时，尚未调用模型。",
    "stage_timeout": "本阶段超过时间预算，迟到结果不再采用。",
    "workflow_timeout": "评估与修订超过本轮总预算。",
    "request_timeout": "模型接口请求超时。",
    "rate_limit": "模型服务限流，请稍后重试。",
    "connection_error": "无法连接模型服务，请检查网络。",
    "authentication_error": "模型服务认证失败，请检查密钥配置。",
    "permission_error": "当前账号无模型调用权限。",
    "model_not_found": "模型或接口路径不存在。",
    "invalid_request": "模型服务不接受本次请求参数。",
    "output_limit": "模型输出被截断，未采用不完整的评价。",
    "invalid_response": "模型返回结构、原文绑定或评价依据不完整。",
    "invalid_json": "评估结果不是完整的 JSON，未采用无法解析的内容。",
    "invalid_binding": "部分评价的候选编号或版本不匹配，未将其应用到原文。",
    "incomplete_evidence": "评价缺少有效维度、依据或整组判断；未通过校验的评价不采用。",
    "provider_error": "模型服务返回错误。",
    "model_error": "模型调用未完成。",
    "internal_error": "评估任务内部异常，请查看服务端错误类型。",
}


def _validation_error_code(errors):
    kinds = {error.get("type") for error in errors}
    if "invalid_or_empty_json_object" in kinds:
        return "invalid_json"
    if kinds & {"unexpected_candidate_id", "duplicate_candidate_id", "batch_binding_mismatch",
                "original_text_mismatch", "unexpected_fields"}:
        return "invalid_binding"
    if kinds & {"missing_or_blank_evidence", "missing_or_invalid_score", "missing_target",
                "missing_or_invalid_group_evidence", "missing_or_invalid_list", "wrong_item_count"}:
        return "incomplete_evidence"
    return "invalid_response"


@dataclass
class _Job:
    owner: str
    payload: dict | None
    texts: list[str]
    created: float
    deadline: float
    fingerprint: str
    timeout: float
    total_budget: float
    parent_request_id: str | None = None
    review_mode: str = "full"
    status: str = "pending"
    phase: str = "queued"
    started: float | None = None
    phase_started: float | None = None
    total_deadline: float | None = None
    finished: float | None = None
    quality: ReplyQualityReport | None = None
    trace: object | None = None
    error_code: str | None = None
    repair_status: str = "not_needed"
    revision_replies: list | None = None
    revision_quality: ReplyQualityReport | None = None
    future: Future | None = None


class QualityJobManager:
    def __init__(self, *, workers=2, max_active=16, max_retained=64, ttl=600, clock=monotonic):
        self._lock = RLock()
        self._jobs: dict[str, _Job] = {}
        self._executor = None
        self._workers, self._max_active, self._max_retained = workers, max_active, max_retained
        self._ttl, self._clock = ttl, clock
        self._stop = Event()
        self._reaper = None

    def start(self):
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="quality-review")
                self._stop.clear()
                self._reaper = Thread(target=self._expire_jobs, name="quality-expiry", daemon=True)
                self._reaper.start()

    def _expire_jobs(self):
        while not self._stop.wait(1):
            with self._lock:
                self._prune()

    def stop(self):
        with self._lock:
            self._stop.set()
            reaper, self._reaper = self._reaper, None
            executor, self._executor = self._executor, None
            for job in self._jobs.values():
                self._finish(job, "cancelled")
                if job.future:
                    job.future.cancel()
            self._jobs.clear()
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        if reaper:
            reaper.join()

    def _finish(self, job, status):
        job.status, job.payload, job.finished = status, None, self._clock()
        if status == "cancelled":
            job.quality = job.revision_quality = job.revision_replies = None
            job.texts = []

    def _fail(self, job, code):
        job.error_code = code if code in ERRORS else "model_error"
        timeout = code in {"queue_timeout", "stage_timeout", "workflow_timeout", "request_timeout"}
        if job.quality is not None:
            job.repair_status = "timed_out" if timeout else "failed"
            self._finish(job, "completed")
        else:
            self._finish(job, "timed_out" if timeout else "failed")

    def _prune(self):
        now = self._clock()
        for job_id, job in list(self._jobs.items()):
            if job.status in ACTIVE and now >= job.deadline:
                code = "queue_timeout" if job.phase == "queued" else (
                    "workflow_timeout" if now >= job.total_deadline else "stage_timeout")
                self._fail(job, code)
                if job.future:
                    job.future.cancel()
            if (job.finished is not None and now - job.finished >= self._ttl
                    and (not job.future or job.future.done())):
                del self._jobs[job_id]

    def submit(self, user_id, payload, timeout):
        serialized = json.dumps(payload, default=lambda value: value.model_dump(), sort_keys=True, ensure_ascii=False)
        fingerprint = sha256(serialized.encode()).hexdigest()
        with self._lock:
            self._prune()
            if self._executor is None:
                return None, "unavailable"
            for key, job in self._jobs.items():
                if job.owner == user_id and job.fingerprint == fingerprint and job.future and not job.future.done():
                    if job.status == "cancelled":
                        return None, "busy"
                    return key, job.status
            if sum(job.status in ACTIVE or bool(job.future and not job.future.done())
                   for job in self._jobs.values()) >= self._max_active:
                return None, "busy"
            while len(self._jobs) >= self._max_retained:
                terminal = next((key for key, job in self._jobs.items()
                    if job.status not in ACTIVE and (not job.future or job.future.done())), None)
                if terminal is None:
                    return None, "busy"
                del self._jobs[terminal]
            settings, now, job_id = get_settings(), self._clock(), uuid4().hex
            snapshot = deepcopy(payload)
            parent = current_trace.get()
            job = _Job(user_id, snapshot, [r.text for r in snapshot["replies"]], now,
                now + settings.quality_queue_timeout_seconds, fingerprint, timeout,
                settings.quality_total_timeout_seconds, parent.request_id if parent else None)
            job.review_mode = snapshot.get("review_mode", "full")
            self._jobs[job_id] = job
            try:
                job.future = self._executor.submit(self._run, job_id)
            except RuntimeError:
                del self._jobs[job_id]
                return None, "unavailable"
            return job_id, "pending"

    def _diagnostics(self, job):
        end = job.finished if job.finished is not None else self._clock()
        start = job.started if job.started is not None else end
        snapshot = job.trace.snapshot() if job.trace else {"stages": [], "model_calls": []}
        return {**snapshot, "parent_request_id": job.parent_request_id,
            "queue_wait_ms": round(max(0, start - job.created) * 1000, 2),
            "total_ms": round(max(0, end - start) * 1000, 2),
            "elapsed_ms": round(max(0, end - job.created) * 1000, 2),
            "phase_elapsed_ms": round(max(0, end - (job.phase_started if job.phase_started is not None else job.created)) * 1000, 2),
            "phase": job.phase, "review_mode": job.review_mode,
            "phase_review_mode": job.review_mode if job.phase in {"queued", "review"} else "full",
            "error_code": job.error_code, "error_message": ERRORS.get(job.error_code, ""),
            "worker_active": bool(job.future and not job.future.done()),
            "model_timeout_seconds": job.timeout, "total_budget_seconds": job.total_budget}

    def get(self, job_id, user_id):
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            if job is None or job.owner != user_id:
                return None
            message = MESSAGES[job.status]
            if job.status == "running":
                message = {"review": "正在评估原文，不会修改输入框。", "repair": "原文存在具体问题，正在生成独立改进版。",
                           "recheck": "正在复评改进版；未通过前不当作合格答案。"}[job.phase]
            elif job.repair_status in {"failed", "timed_out"}:
                message = "原文评估已完成，但改进版未完成验证；保留原稿，不自动重试。"
            elif job.repair_status == "completed":
                message = "已完成原文评估及一次修订复评；改进版单独展示，请确认后使用。"
            elif job.repair_status == "unverified":
                message = "已修订一次，但改进版仍未通过复评；需要人工确认，不继续循环。"
            elif job.quality and job.quality.status == "partial":
                count = sum(item.status == "reviewed" for item in job.quality.items)
                message = f"已保留 {count}/{len(job.texts)} 条有效评价；本轮校验未全部完成，不自动修订或重试。"
            return QualityJobResponse(job_id=job_id, status=job.status, candidate_texts=list(job.texts),
                quality=job.quality.model_copy(deep=True) if job.quality else None, message=message,
                diagnostics=self._diagnostics(job), phase=job.phase, repair_status=job.repair_status,
                revision_replies=deepcopy(job.revision_replies or []),
                revision_quality=job.revision_quality.model_copy(deep=True) if job.revision_quality else None)

    def cancel(self, job_id, user_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.owner != user_id:
                return False
            self._finish(job, "cancelled")
            if job.future:
                job.future.cancel()
            return True

    def _begin_stage(self, job_id, job, phase):
        with self._lock:
            self._prune()
            if self._jobs.get(job_id) is not job or job.status not in ACTIVE:
                return False
            now = self._clock()
            if job.started is None:
                job.started = now
                job.total_deadline = now + job.total_budget
            job.status, job.phase, job.phase_started = "running", phase, now
            job.deadline = min(now + job.timeout, job.total_deadline)
            if phase != "review":
                job.repair_status = "running"
            self._prune()
            return job.status == "running"

    def _review(self, job, payload, stage):
        with model_request_timeout(max(.01, job.deadline - self._clock())), quality_model_limits(payload.get("review_mode", "full")):
            result = timed_call(stage, review_candidates, **payload,
                targets={i: ["检查事实与承诺、情绪分寸、关系及候选差异"]
                         for i, reply in enumerate(payload["replies"]) if reply.source == "model"},
                audit_only=True, include_scores=False, raise_errors=True)
        if ([reply.text for reply in result.replies] != [reply.text for reply in payload["replies"]]
                or len(result.report.items) != len(payload["replies"])
                or not any(item.status == "reviewed" for item in result.report.items)
                or stage == "quality.recheck" and result.report.status != "reviewed"):
            code = _validation_error_code(job.trace.validation_errors if job.trace else [])
            raise ModelCallError("无完整有效的原文评价。", kind=code)
        if any(item.status == "reviewed" and item.evaluated_text != reply.text
               for item, reply in zip(result.report.items, payload["replies"])):
            raise ModelCallError("无有效原文评价。", kind="invalid_response")
        return result.report

    def _run(self, job_id):
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            if job is None or job.status != "pending":
                return
            payload, job.payload = job.payload, None
        try:
            with trace_scope() as trace, quality_model_limits():
                job.trace = trace
                if not self._begin_stage(job_id, job, "review"):
                    return
                report = self._review(job, payload, "quality.background")
                with self._lock:
                    self._prune()
                    if job.status != "running":
                        return
                    job.quality = report
                    if report.status == "partial":
                        job.error_code = _validation_error_code(trace.validation_errors) if trace.validation_errors else None
                        self._finish(job, "completed")
                        return
                settings = get_settings()
                targets = repair_targets(report, payload["replies"], settings.quality_repair_threshold)
                if settings.quality_repair_enabled and targets:
                    if not self._begin_stage(job_id, job, "repair"):
                        return
                    with model_request_timeout(max(.01, job.deadline - self._clock())):
                        revised = timed_call("quality.repair", repair_candidates, payload, report, targets)
                    if not self._begin_stage(job_id, job, "recheck"):
                        return
                    reviewed = self._review(job, {**payload, "replies": revised, "review_mode": "full"}, "quality.recheck")
                    with self._lock:
                        self._prune()
                        if job.status != "running":
                            return
                        job.revision_replies, job.revision_quality = revised, reviewed
                        passed = reviewed.passed and not repair_targets(reviewed, revised, settings.quality_repair_threshold)
                        job.repair_status = "completed" if passed else "unverified"
                with self._lock:
                    self._prune()
                    if job.status == "running":
                        self._finish(job, "completed")
        except Exception as exc:
            with self._lock:
                self._prune()
                if self._jobs.get(job_id) is job and job.status in ACTIVE:
                    self._fail(job, exc.kind if isinstance(exc, ModelCallError) else "internal_error")
            logger.warning("quality_job_error job_id=%s error_type=%s", job_id, type(exc).__name__)
        finally:
            payload = None
            with self._lock:
                logger.info("quality_job_finished %s", json.dumps({"job_id": job_id, "status": job.status,
                    "repair_status": job.repair_status, **self._diagnostics(job)}, ensure_ascii=True))


quality_jobs = QualityJobManager()


def select_review_mode(analysis, scenario, controls, chat, weather_evidence=()) -> str:
    if not get_settings().quality_quick_enabled or weather_evidence or requires_quality_gate(analysis, scenario, controls, chat):
        return "full"
    return "quick"


def enqueue_quality(*, user_id: str, chat_text: str, replies: list[ReplyCandidate], relation_rules: dict,
                    memory_summary: str, style_summary: str, controls: ReplyControls,
                    style_evidence_available: bool = False, review_mode: str = "full") -> QualityJobResponse:
    payload = dict(chat_text=chat_text, replies=replies, relation_rules=relation_rules,
                   memory_summary=memory_summary, style_summary=style_summary, controls=controls,
                   style_evidence_available=style_evidence_available, review_mode=review_mode)
    initial = review_candidates(**payload, targets={}).report
    if not any(reply.source == "model" for reply in replies):
        job_id, status = None, "disabled"
    elif len(json.dumps(payload, default=lambda value: value.model_dump(), ensure_ascii=False).encode()) > 262144:
        job_id, status = None, "unavailable"
    else:
        job_id, status = quality_jobs.submit(user_id, payload, get_settings().quality_timeout_seconds)
    # Reuse an identical active job, including a timed-out request still unwinding.
    if job_id and status != "pending":
        existing = quality_jobs.get(job_id, user_id)
        if existing:
            existing.quality = existing.quality or initial
            return existing
    initial.summary = MESSAGES[status]
    initial.status = "pending" if status == "pending" else "not_reviewed"
    if status == "pending":
        for item, reply in zip(initial.items, replies):
            if reply.source == "model":
                item.status = "pending"
    return QualityJobResponse(job_id=job_id, status=status, quality=initial,
        candidate_texts=[reply.text for reply in replies], message=MESSAGES[status])
