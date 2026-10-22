import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from chat_reply_agent.config import get_settings
from chat_reply_agent.database import get_db, init_db
from chat_reply_agent.schemas import (
    ContactListResponse,
    ContactMemoryDeleteResponse,
    ContactMemorySnapshot,
    ContactMemoryUpdateRequest,
    CorrectedReplyRequest,
    EvalCaseListResponse,
    EvalRunRequest,
    EvalRunResponse,
    ExtractedChat,
    MemoryEpisodeDeleteResponse,
    MemorySearchResponse,
    MemoryRetrievalDiagnostics,
    PreparedReplyRequest,
    RegenerateReplyRequest,
    RegenerateReplyResponse,
    ChatScenario,
    ReplyControls,
    ReplyFeedbackRequest,
    ReplyFeedbackResponse,
    ReplyResponse,
    QualityReviewRequest,
    QualityJobResponse,
    RetrievedMemoryItem,
    StyleExampleCreateRequest,
    StyleExampleDeleteResponse,
    StyleExampleItem,
    StyleExamplesResponse,
    StyleEvidence,
    StyleResetResponse,
)
from chat_reply_agent.services.generation_service import understand_and_generate
from chat_reply_agent.services.upload_service import MAX_BATCH_BYTES, save_screenshot
from chat_reply_agent.services.embedding_service import embedding_runtime_status
from chat_reply_agent.services.quality_queue_service import enqueue_quality, quality_jobs
from chat_reply_agent.services.quality_repair_service import requires_quality_gate
from chat_reply_agent.services.memory_queue_service import enqueue_memory, start_memory_worker, stop_memory_worker
from chat_reply_agent.models import MemoryJob, ConversationMemory
from chat_reply_agent.services.contact_memory_service import (
    get_contact_context,
    get_contact_snapshot,
    list_contact_snapshots,
    update_contact_memory_manually,
)
from chat_reply_agent.services.context_service import (
    build_context_retrieval_query,
    build_model_chat_context,
    get_recent_chat_text,
    normalize_background_context,
)
from chat_reply_agent.services.correction_service import save_chat_correction
from chat_reply_agent.services.data_management_service import (
    delete_contact_memory_data,
    delete_memory_episode_data,
    reset_style_data,
)
from chat_reply_agent.services.feedback_service import (
    get_reply_preference_summary,
    save_reply_feedback,
)
from chat_reply_agent.services.evaluation_service import (
    list_evaluation_cases,
    run_evaluation,
)
from chat_reply_agent.services.regression_service import DATASET_PATH, compare_reply_regressions
from chat_reply_agent.services.reply_service import (
    regenerate_reply_candidate,
)
from chat_reply_agent.services.retrieval_service import (
    build_retrieval_context,
    list_memory_episodes,
    retrieve_relevant_memories,
)
from chat_reply_agent.services.skill_loader import load_relation_skill
from chat_reply_agent.services.style_example_service import (
    create_manual_style_example,
    delete_style_example,
    get_style_examples,
    get_style_examples_summary,
    save_style_example_from_feedback,
)
from chat_reply_agent.services.style_service import (
    UserStyleSnapshot,
    build_current_user_style,
    get_saved_style_snapshot,
)
from chat_reply_agent.services.timing_service import suggest_reply_timing
from chat_reply_agent.services.telemetry_service import (
    log_trace, timed_call, timed_stage, trace_scope, traced_response,
)
from chat_reply_agent.services.llm_client import ModelCallError, has_llm_config, configured_models, weather_tools
from chat_reply_agent.services.weather_contract import weather_evidence_context
from chat_reply_agent.services.vision_service import (
    extract_chat_from_image,
    merge_extracted_chats,
)


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.22.0")
MAX_SCREENSHOT_COUNT = 6


@app.middleware("http")
async def record_request_timing(request, call_next):
    with trace_scope() as trace:
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = trace.request_id
            response.headers["Server-Timing"] = f"total;dur={trace.snapshot()['total_ms']}"
            return response
        finally:
            log_trace(trace, status_code)


@dataclass
class _GenerationContext:
    memory_context: str
    style_context: str
    contact_memory: ContactMemorySnapshot
    current_style: UserStyleSnapshot
    saved_style: UserStyleSnapshot
    preference_summary: str
    style_examples_summary: str
    retrieved_memories: list[RetrievedMemoryItem]
    retrieval: MemoryRetrievalDiagnostics


@app.on_event("startup")
def on_startup() -> None:
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    init_db()
    start_memory_worker()
    quality_jobs.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    quality_jobs.stop()
    stop_memory_worker()


@app.get("/")
def index() -> FileResponse:
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(index_path)


@app.get("/evals")
def evals_page() -> FileResponse:
    page_path = Path(__file__).resolve().parent / "static" / "evals.html"
    return FileResponse(page_path)


@app.get("/memory")
def memory_page() -> FileResponse:
    page_path = Path(__file__).resolve().parent / "static" / "memory.html"
    return FileResponse(page_path)


@app.get("/api/status")
def status() -> dict:
    return {
        "app": settings.app_name,
        "version": app.version,
        "llm_configured": has_llm_config(),
        "text_model": configured_models()["strong"],
        "vision_model": settings.openai_vision_model_name,
        "models": configured_models(),
        "weather_tool_enabled": settings.weather_tool_enabled,
        "memory_retrieval": {
            "backend": settings.memory_similarity_backend,
            "embedding_provider": settings.memory_embedding_provider,
            "embedding_model": settings.memory_embedding_model,
            **embedding_runtime_status(settings),
        },
    }


@app.get("/api/memory-jobs/{job_id}")
def memory_job_status(job_id: str, user_id: str = "default", db: Session = Depends(get_db)) -> dict:
    job = db.scalar(select(MemoryJob).where(MemoryJob.id == job_id, MemoryJob.user_id == user_id))
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或已删除。")
    result = json.loads(job.result_json)
    response = {"job_id": job.id, "status": job.status, "memory_persisted": job.status == "completed",
                "error": "后台记忆更新失败，本轮未保存；可重新生成后再试。" if job.error else "", **result}
    if job.status == "completed":
        memory = db.get(ConversationMemory, result.get("saved_memory_id"))
        if memory is None or memory.user_id != user_id:
            response.update(status="cancelled", memory_persisted=False, saved_memory_id=None)
        else:
            response.update(memory_summary=memory.summary,
                contact_memory=get_contact_snapshot(db, user_id, job.contact_id).model_dump(),
                style_summary=get_saved_style_snapshot(db, user_id, job.relation_type).summary)
    return response


@app.get("/api/quality-jobs/{job_id}", response_model=QualityJobResponse)
def quality_job_status(job_id: str, user_id: str = "default") -> QualityJobResponse:
    job = quality_jobs.get(job_id, user_id)
    if job is None:
        raise HTTPException(status_code=404, detail="评估任务不存在、已过期或服务已重启。")
    return job


@app.delete("/api/quality-jobs/{job_id}")
def cancel_quality_job(job_id: str, user_id: str = "default") -> dict:
    if not quality_jobs.cancel(job_id, user_id):
        raise HTTPException(status_code=404, detail="评估任务不存在或已过期。")
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/api/quality-jobs", response_model=QualityJobResponse)
def review_current_replies(payload: QualityReviewRequest, db: Session = Depends(get_db)) -> QualityJobResponse:
    try:
        contact_id, contact_name = _normalize_contact_fields(payload.contact_id, payload.contact_name)
        background = normalize_background_context(payload.background_context)
        chat_text = build_model_chat_context(payload.extracted_chat, background)
        context = _load_generation_context(db, payload.user_id, payload.relation_type, contact_id,
                                          contact_name, payload.extracted_chat, ChatScenario(), background)
        return enqueue_quality(user_id=payload.user_id,
            chat_text=chat_text + weather_evidence_context(payload.weather_evidence), replies=payload.replies,
            relation_rules=load_relation_skill(payload.relation_type), memory_summary=context.memory_context,
            style_summary=context.style_context, controls=payload.controls,
            style_evidence_available=bool(context.current_style.summary or context.saved_style.summary
                                          or context.style_examples_summary))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelCallError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.delete("/api/memory-jobs/{job_id}")
def cancel_memory_job(job_id: str, user_id: str = "default", db: Session = Depends(get_db)) -> dict:
    from sqlalchemy import delete
    result = db.execute(delete(MemoryJob).where(MemoryJob.id == job_id, MemoryJob.user_id == user_id,
                                              MemoryJob.status != "completed"))
    db.commit()
    if not result.rowcount:
        raise HTTPException(status_code=404, detail="没有可取消的任务；已保存的记忆请在管理页删除。")
    return {"deleted": True}


@app.get("/api/evals/cases", response_model=EvalCaseListResponse)
def evaluation_cases(dataset: Literal["core", "v020"] = "core") -> EvalCaseListResponse:
    return list_evaluation_cases(DATASET_PATH) if dataset == "v020" else list_evaluation_cases()


@app.get("/api/evals/regressions")
def reply_regression_comparison() -> dict:
    return compare_reply_regressions()


@app.post("/api/evals/run", response_model=EvalRunResponse)
def run_evaluation_cases(payload: EvalRunRequest) -> EvalRunResponse:
    try:
        return run_evaluation(payload.case_ids, DATASET_PATH) if payload.dataset == "v020" else run_evaluation(payload.case_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/vision/extract", response_model=ExtractedChat)
async def extract_chat_preview(image: UploadFile = File(...)) -> ExtractedChat:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件。")
    image_path = await _save_upload(image)
    try:
        return await run_in_threadpool(extract_chat_from_image, image_path)
    except ModelCallError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        _remove_upload(image_path)


@app.post("/api/vision/extract-batch", response_model=ExtractedChat)
async def extract_chat_batch(
    images: list[UploadFile] = File(...),
) -> ExtractedChat:
    if not images:
        raise HTTPException(status_code=400, detail="请至少上传一张聊天截图。")
    if len(images) > MAX_SCREENSHOT_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"一次最多上传 {MAX_SCREENSHOT_COUNT} 张截图。",
        )
    for index, image in enumerate(images, start=1):
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"第 {index} 个文件不是图片，请重新选择。",
            )

    image_paths: list[Path] = []
    extracted_chats: list[ExtractedChat] = []
    try:
        total_bytes = 0
        # Validate the entire batch before paying for any vision request.
        for index, image in enumerate(images, start=1):
            image_path = await _save_upload(image)
            image_paths.append(image_path)
            total_bytes += image_path.stat().st_size
            if total_bytes > MAX_BATCH_BYTES:
                raise HTTPException(413, "本次截图总大小不能超过 60 MB，请分批上传。")
        for index, image_path in enumerate(image_paths, start=1):
            try:
                extracted_chat = await run_in_threadpool(
                    extract_chat_from_image,
                    image_path,
                )
            except ModelCallError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"第 {index} 张截图识别失败：{exc}",
                ) from exc
            extracted_chats.append(extracted_chat)
        return merge_extracted_chats(extracted_chats)
    finally:
        for image_path in image_paths:
            _remove_upload(image_path)


@app.post("/api/reply", response_model=ReplyResponse)
async def create_reply_advice(
    relation_type: str = Form(...),
    user_id: str = Form("default"),
    contact_id: str = Form("default-contact"),
    contact_name: str = Form(""),
    reply_intent: str = Form("auto"),
    custom_intent: str = Form(""),
    length_preference: str = Form("short"),
    intimacy_level: int = Form(2),
    initiative_level: int = Form(2),
    save_to_memory: bool = Form(True),
    background_context: str = Form(""),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ReplyResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件。")

    contact_id, contact_name = _normalize_contact_fields(contact_id, contact_name)
    controls = _make_reply_controls(
        intent=reply_intent,
        custom_intent=custom_intent,
        length_preference=length_preference,
        intimacy_level=intimacy_level,
        initiative_level=initiative_level,
    )
    image_path = await _save_upload(image)

    try:
        extracted_chat = await run_in_threadpool(extract_chat_from_image, image_path)
        return await run_in_threadpool(_build_reply_response,
            db=db,
            user_id=user_id,
            relation_type=relation_type,
            contact_id=contact_id,
            contact_name=contact_name,
            extracted_chat=extracted_chat,
            controls=controls,
            save_to_memory=save_to_memory,
            background_context=background_context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelCallError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        _remove_upload(image_path)


@app.post("/api/reply/corrected", response_model=ReplyResponse)
def create_reply_from_correction(
    payload: CorrectedReplyRequest,
    db: Session = Depends(get_db),
) -> ReplyResponse:
    contact_id, contact_name = _normalize_contact_fields(
        payload.contact_id,
        payload.contact_name,
    )
    try:
        background_context = normalize_background_context(
            payload.background_context
        )
        extracted_chat = _normalize_corrected_chat(
            payload.extracted_chat,
            allow_empty=bool(background_context),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    correction = None
    if payload.save_to_memory and extracted_chat.messages:
        correction = save_chat_correction(
            db=db,
            user_id=payload.user_id,
            relation_type=payload.relation_type,
            extracted_chat=extracted_chat,
        )

    try:
        response = _build_reply_response(
            db=db,
            user_id=payload.user_id,
            relation_type=payload.relation_type,
            contact_id=contact_id,
            contact_name=contact_name,
            extracted_chat=extracted_chat,
            controls=payload.controls,
            save_to_memory=payload.save_to_memory,
            background_context=background_context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelCallError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response.saved_correction_id = correction.id if correction else None
    return response


@app.post("/api/reply/from-chat", response_model=ReplyResponse)
def create_reply_from_confirmed_chat(
    payload: PreparedReplyRequest,
    db: Session = Depends(get_db),
) -> ReplyResponse:
    contact_id, contact_name = _normalize_contact_fields(
        payload.contact_id,
        payload.contact_name,
    )
    try:
        background_context = normalize_background_context(
            payload.background_context
        )
        has_chat_messages = bool(payload.extracted_chat.messages)
        extracted_chat = _normalize_confirmed_chat(
            payload.extracted_chat,
            marker=(
                "用户已确认截图识别内容。"
                if has_chat_messages
                else "用户仅提供文字背景，未上传聊天截图。"
            ),
            allow_empty=bool(background_context),
        )
        return _build_reply_response(
            db=db,
            user_id=payload.user_id,
            relation_type=payload.relation_type,
            contact_id=contact_id,
            contact_name=contact_name,
            extracted_chat=extracted_chat,
            controls=payload.controls,
            save_to_memory=payload.save_to_memory,
            background_context=background_context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelCallError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(
    "/api/reply/regenerate",
    response_model=RegenerateReplyResponse,
)
@traced_response
def regenerate_single_reply(
    payload: RegenerateReplyRequest,
    db: Session = Depends(get_db),
) -> RegenerateReplyResponse:
    contact_id, contact_name = _normalize_contact_fields(
        payload.contact_id,
        payload.contact_name,
    )
    try:
        background_context = normalize_background_context(
            payload.background_context
        )
        chat_text = build_model_chat_context(
            payload.extracted_chat,
            background_context,
        )
        relation_rules = load_relation_skill(payload.relation_type)
        context = _load_generation_context(
            db=db,
            user_id=payload.user_id,
            relation_type=payload.relation_type,
            contact_id=contact_id,
            contact_name=contact_name,
            extracted_chat=payload.extracted_chat,
            scenario=payload.scenario,
            background_context=background_context,
        )
        with weather_tools() as weather_evidence:
            reply = timed_call("reply.regenerate", regenerate_reply_candidate,
                chat_text=chat_text,
                analysis=payload.analysis,
                scenario=payload.scenario,
                relation_rules=relation_rules,
                memory_summary=context.memory_context,
                style_summary=context.style_context,
                controls=payload.controls,
                existing_replies=payload.existing_replies,
                candidate_index=payload.candidate_index,
            )
        existing = list(payload.existing_replies)
        existing[payload.candidate_index] = reply
        review_job = timed_call("quality.enqueue", enqueue_quality,
            user_id=payload.user_id,
            chat_text=chat_text + weather_evidence_context((payload.weather_evidence + weather_evidence)[-6:]),
            replies=existing,
            relation_rules=relation_rules,
            memory_summary=context.memory_context,
            style_summary=context.style_context,
            style_evidence_available=bool(context.current_style.summary or context.saved_style.summary
                                          or context.style_examples_summary),
            controls=payload.controls,
        )
        final_reply = reply
        quality = review_job.quality.items[payload.candidate_index]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelCallError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return RegenerateReplyResponse(
        retrieval=context.retrieval,
        weather_evidence=weather_evidence,
        candidate_index=payload.candidate_index,
        reply=final_reply,
        quality=quality,
        quality_report=review_job.quality,
        quality_job_id=review_job.job_id,
        quality_status=review_job.status,
        quality_job=review_job,
        quality_gate_required=requires_quality_gate(payload.analysis, payload.scenario, payload.controls, payload.extracted_chat),
    )


@app.post("/api/reply/feedback", response_model=ReplyFeedbackResponse)
def create_reply_feedback(
    payload: ReplyFeedbackRequest,
    db: Session = Depends(get_db),
) -> ReplyFeedbackResponse:
    feedback = save_reply_feedback(db, payload)
    save_style_example_from_feedback(db, feedback)
    preference_summary = get_reply_preference_summary(
        db=db,
        user_id=payload.user_id,
        relation_type=payload.relation_type,
    )
    style_examples_summary = get_style_examples_summary(
        db=db,
        user_id=payload.user_id,
        relation_type=payload.relation_type,
    )
    return ReplyFeedbackResponse(
        feedback_id=feedback.id,
        preference_summary=preference_summary,
        style_examples_summary=style_examples_summary,
    )


@app.get("/api/style/examples", response_model=StyleExamplesResponse)
def list_style_examples(
    relation_type: str,
    user_id: str = "default",
    limit: int = Query(default=30, ge=1, le=50),
    db: Session = Depends(get_db),
) -> StyleExamplesResponse:
    examples = get_style_examples(db, user_id, relation_type, limit)
    summary = get_style_examples_summary(db, user_id, relation_type)
    automatic_style = get_saved_style_snapshot(db, user_id, relation_type)
    return StyleExamplesResponse(
        examples=[
            StyleExampleItem(
                id=example.id,
                text=example.text,
                source=example.source,
                created_at=example.created_at,
            )
            for example in examples
        ],
        summary=summary,
        automatic_summary=automatic_style.summary,
        automatic_examples=list(automatic_style.phrases),
    )


@app.post("/api/style/examples", response_model=StyleExampleItem, status_code=201)
def add_style_example(
    payload: StyleExampleCreateRequest,
    db: Session = Depends(get_db),
) -> StyleExampleItem:
    example = create_manual_style_example(
        db=db,
        user_id=payload.user_id,
        relation_type=payload.relation_type,
        text=payload.text,
    )
    return StyleExampleItem(
        id=example.id,
        text=example.text,
        source=example.source,
        created_at=example.created_at,
    )


@app.delete(
    "/api/style/examples/{example_id}",
    response_model=StyleExampleDeleteResponse,
)
def remove_style_example(
    example_id: int,
    relation_type: str,
    user_id: str = "default",
    db: Session = Depends(get_db),
) -> StyleExampleDeleteResponse:
    deleted = delete_style_example(
        db=db,
        user_id=user_id,
        relation_type=relation_type,
        example_id=example_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="未找到这条风格样本。")
    return StyleExampleDeleteResponse(example_id=example_id)


@app.delete("/api/style/data", response_model=StyleResetResponse)
def clear_style_data(
    relation_type: str,
    user_id: str = "default",
    db: Session = Depends(get_db),
) -> StyleResetResponse:
    return reset_style_data(
        db=db,
        user_id=user_id,
        relation_type=relation_type,
    )


@app.get("/api/contacts", response_model=ContactListResponse)
def list_contacts(
    user_id: str = "default",
    db: Session = Depends(get_db),
) -> ContactListResponse:
    return ContactListResponse(contacts=list_contact_snapshots(db, user_id))


@app.get(
    "/api/contacts/{contact_id}/memory",
    response_model=ContactMemorySnapshot,
)
def read_contact_memory(
    contact_id: str,
    user_id: str = "default",
    db: Session = Depends(get_db),
) -> ContactMemorySnapshot:
    normalized_id, _ = _normalize_contact_fields(contact_id, "")
    snapshot = get_contact_snapshot(db, user_id, normalized_id)
    if not snapshot.has_previous_memory:
        raise HTTPException(status_code=404, detail="尚未保存该联系人的记忆。")
    return snapshot


@app.put(
    "/api/contacts/{contact_id}/memory",
    response_model=ContactMemorySnapshot,
)
def edit_contact_memory(
    contact_id: str,
    payload: ContactMemoryUpdateRequest,
    db: Session = Depends(get_db),
) -> ContactMemorySnapshot:
    normalized_id, _ = _normalize_contact_fields(contact_id, "")
    snapshot = update_contact_memory_manually(
        db=db,
        user_id=payload.user_id,
        contact_id=normalized_id,
        payload=payload,
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="尚未保存该联系人的记忆。")
    return snapshot


@app.delete(
    "/api/contacts/{contact_id}/memory",
    response_model=ContactMemoryDeleteResponse,
)
def remove_contact_memory(
    contact_id: str,
    user_id: str = "default",
    db: Session = Depends(get_db),
) -> ContactMemoryDeleteResponse:
    normalized_id, _ = _normalize_contact_fields(contact_id, "")
    response = delete_contact_memory_data(
        db=db,
        user_id=user_id,
        contact_id=normalized_id,
    )
    if not response.deleted:
        raise HTTPException(status_code=404, detail="尚未保存该联系人的记忆。")
    return response


@app.get(
    "/api/contacts/{contact_id}/memories",
    response_model=MemorySearchResponse,
)
def search_contact_memories(
    contact_id: str,
    user_id: str = "default",
    query: str = "",
    limit: int = Query(default=10, ge=1, le=10),
    db: Session = Depends(get_db),
) -> MemorySearchResponse:
    normalized_id, _ = _normalize_contact_fields(contact_id, "")
    normalized_query = query.strip()[:1000]
    retrieval = MemoryRetrievalDiagnostics(message="历史列表未执行相似度检索。")
    if normalized_query:
        try:
            memories = retrieve_relevant_memories(
                db=db,
                user_id=user_id,
                contact_id=normalized_id,
                query_text=normalized_query,
                limit=limit,
                diagnostics=retrieval,
            )
        except ModelCallError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    else:
        memories = list_memory_episodes(
            db=db,
            user_id=user_id,
            contact_id=normalized_id,
            limit=limit,
        )
    return MemorySearchResponse(
        contact_id=normalized_id,
        query=normalized_query,
        memories=memories,
        retrieval=retrieval,
    )


@app.delete(
    "/api/contacts/{contact_id}/memories/{memory_id}",
    response_model=MemoryEpisodeDeleteResponse,
)
def remove_contact_memory_episode(
    contact_id: str,
    memory_id: int,
    user_id: str = "default",
    db: Session = Depends(get_db),
) -> MemoryEpisodeDeleteResponse:
    normalized_id, _ = _normalize_contact_fields(contact_id, "")
    deleted = delete_memory_episode_data(
        db=db,
        user_id=user_id,
        contact_id=normalized_id,
        memory_id=memory_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="未找到这条历史片段。")
    return MemoryEpisodeDeleteResponse(
        contact_id=normalized_id,
        memory_id=memory_id,
    )


@traced_response
def _build_reply_response(
    db: Session,
    user_id: str,
    relation_type: str,
    contact_id: str,
    contact_name: str,
    extracted_chat: ExtractedChat,
    controls: ReplyControls | None = None,
    save_to_memory: bool = True,
    background_context: str = "",
) -> ReplyResponse:
    controls = controls or ReplyControls()
    relation_rules = load_relation_skill(relation_type)
    background_context = normalize_background_context(background_context)
    screenshot_text = get_recent_chat_text(extracted_chat)
    chat_text = build_model_chat_context(extracted_chat, background_context)
    context = _load_generation_context(
        db=db,
        user_id=user_id,
        relation_type=relation_type,
        contact_id=contact_id,
        contact_name=contact_name,
        extracted_chat=extracted_chat,
        scenario=ChatScenario(),
        background_context=background_context,
    )
    generated = timed_call("reply.understand_generate", understand_and_generate,
        chat_text=chat_text,
        relation_rules=relation_rules,
        memory_summary=context.memory_context,
        style_summary=context.style_context,
        controls=controls,
    )
    analysis, scenario, replies = generated.analysis, generated.scenario, generated.replies
    extracted_chat = extracted_chat.model_copy(update={
        "topic": generated.topic,
        "topic_source": "context" if generated.topic else "pending",
    })
    strategy = generated.strategy
    timing = suggest_reply_timing(analysis, scenario, relation_rules)
    review_job = timed_call("quality.enqueue", enqueue_quality,
        user_id=user_id,
        chat_text=chat_text + weather_evidence_context(generated.weather_evidence),
        replies=replies,
        relation_rules=relation_rules,
        memory_summary=context.memory_context,
        style_summary=context.style_context,
        style_evidence_available=bool(context.current_style.summary or context.saved_style.summary
                                      or context.style_examples_summary),
        controls=controls,
    )

    contact_memory = context.contact_memory
    memory_summary = contact_memory.summary
    profile_style = context.saved_style
    style_summary = context.current_style.summary or profile_style.summary
    saved_memory_id = None
    memory_job_id = None
    if save_to_memory and any(reply.source == "model" for reply in replies):
        memory_job_id = timed_call("memory.enqueue", enqueue_memory, db,
            user_id=user_id, contact_id=contact_id, contact_name=contact_name,
            relation_type=relation_type, chat_text=chat_text,
            analysis=analysis.model_dump(), scenario=scenario.model_dump(),
            extracted_chat=extracted_chat.model_dump())

    return ReplyResponse(
        retrieval=context.retrieval,
        weather_evidence=generated.weather_evidence,
        user_id=user_id,
        relation_type=relation_type,
        controls=controls,
        memory_persisted=False,
        memory_job_id=memory_job_id,
        memory_status="pending" if memory_job_id else "disabled",
        contact_id=contact_id,
        contact_name=contact_memory.contact_name,
        extracted_chat=extracted_chat,
        background_context=background_context,
        screenshot_text=screenshot_text,
        analysis=analysis,
        scenario=scenario,
        timing=timing,
        strategy=strategy,
        replies=replies,
        quality=review_job.quality,
        quality_job_id=review_job.job_id,
        quality_status=review_job.status,
        quality_job=review_job,
        quality_gate_required=requires_quality_gate(analysis, scenario, controls, extracted_chat),
        memory_summary=memory_summary,
        style_summary=style_summary,
        style_evidence=StyleEvidence(
            current_user_message_count=context.current_style.message_count,
            current_summary=context.current_style.summary,
            current_examples=list(context.current_style.phrases[:6]),
            profile_summary=profile_style.summary,
            profile_examples=list(profile_style.phrases[:6]),
        ),
        preference_summary=context.preference_summary,
        style_examples_summary=context.style_examples_summary,
        retrieved_memories=context.retrieved_memories,
        contact_memory=contact_memory,
        saved_memory_id=saved_memory_id,
    )


def _normalize_corrected_chat(
    extracted_chat: ExtractedChat,
    allow_empty: bool = False,
) -> ExtractedChat:
    return _normalize_confirmed_chat(
        extracted_chat,
        marker="用户已人工校正说话人或文本。",
        allow_empty=allow_empty,
    )


def _normalize_confirmed_chat(
    extracted_chat: ExtractedChat,
    marker: str,
    allow_empty: bool = False,
) -> ExtractedChat:
    last_other = ""
    messages = []
    for message in extracted_chat.messages:
        text = message.text.strip()
        if not text:
            continue
        speaker = (
            message.speaker
            if message.speaker in {"me", "other", "unknown"}
            else "unknown"
        )
        side = (
            message.side
            if message.side in {"left", "right", "center", "unknown"}
            else "unknown"
        )
        if speaker == "me":
            side = "right"
        elif speaker == "other":
            side = "left"
        known_speaker = speaker in {"me", "other"}
        confirmed = message.model_copy(
            update={
                "speaker": speaker,
                "side": side,
                "text": text,
                "text_confidence": 1.0,
                "speaker_confidence": 1.0 if known_speaker else 0.0,
                "speaker_source": "user_confirmed" if known_speaker else "unknown",
                "needs_review": not known_speaker,
                "warnings": [] if known_speaker else ["用户尚未确定说话人，请勿按本人或对方推断。"],
            }
        )
        messages.append(confirmed)
        if speaker == "other":
            last_other = text

    if not messages and not allow_empty:
        raise ValueError("请上传聊天截图或填写补充聊天背景。")

    notes = [
        note
        for note in extracted_chat.notes
        if "条消息需要人工复核" not in note
    ]
    if marker not in notes:
        notes.append(marker)
    review_indexes = [index for index, message in enumerate(messages) if message.needs_review]

    return ExtractedChat(
        messages=messages,
        last_message_from_other=last_other,
        confidence=(
            sum(message.text_confidence * 0.7 + message.speaker_confidence * 0.3 for message in messages)
            / len(messages)
            if messages else 0.0
        ),
        notes=notes,
        review_required=bool(review_indexes),
        review_message_indexes=review_indexes,
        source_image_count=extracted_chat.source_image_count,
        merged_overlap_count=extracted_chat.merged_overlap_count,
    )


def _join_context(*parts: str) -> str:
    return "；".join(part for part in parts if part)


@timed_stage("context.load")
def _load_generation_context(
    db: Session,
    user_id: str,
    relation_type: str,
    contact_id: str,
    contact_name: str,
    extracted_chat: ExtractedChat,
    scenario: ChatScenario,
    background_context: str = "",
) -> _GenerationContext:
    previous_contact_memory = get_contact_snapshot(
        db=db,
        user_id=user_id,
        contact_id=contact_id,
        contact_name=contact_name,
        relation_type=relation_type,
    )
    previous_memory = get_contact_context(previous_contact_memory)
    current_style = build_current_user_style(extracted_chat)
    saved_style = get_saved_style_snapshot(db, user_id, relation_type)
    preference_summary = get_reply_preference_summary(db, user_id, relation_type)
    style_examples_summary = get_style_examples_summary(db, user_id, relation_type)
    style_context = _join_context(
        current_style.prompt_context("当前聊天"),
        saved_style.prompt_context("历史"),
        preference_summary,
        style_examples_summary,
    )
    retrieval_query = build_context_retrieval_query(
        extracted_chat,
        background_context,
    )
    current_context = build_model_chat_context(
        extracted_chat,
        background_context,
    )
    retrieval = MemoryRetrievalDiagnostics()
    retrieved_memories = timed_call("memory.retrieve", retrieve_relevant_memories,
        db=db,
        user_id=user_id,
        contact_id=contact_id,
        query_text=retrieval_query,
        relation_type=relation_type,
        scenario_type=scenario.type,
        limit=3,
        exclude_chat_text=current_context,
        diagnostics=retrieval,
    )
    return _GenerationContext(
        memory_context=build_retrieval_context(
            previous_memory,
            retrieved_memories,
        ),
        style_context=style_context,
        contact_memory=previous_contact_memory,
        current_style=current_style,
        saved_style=saved_style,
        preference_summary=preference_summary,
        style_examples_summary=style_examples_summary,
        retrieved_memories=retrieved_memories,
        retrieval=retrieval,
    )

def _make_reply_controls(
    *,
    intent: str,
    custom_intent: str,
    length_preference: str,
    intimacy_level: int,
    initiative_level: int,
) -> ReplyControls:
    try:
        return ReplyControls(
            intent=intent,
            custom_intent=custom_intent.strip(),
            length_preference=length_preference,
            intimacy_level=intimacy_level,
            initiative_level=initiative_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="回复设定参数无效。") from exc


def _normalize_contact_fields(contact_id: str, contact_name: str) -> tuple[str, str]:
    normalized_id = (contact_id or "").strip()[:64] or "default-contact"
    fallback_name = "默认联系人" if normalized_id == "default-contact" else normalized_id
    normalized_name = (contact_name or "").strip()[:64] or fallback_name
    return normalized_id, normalized_name


async def _save_upload(image: UploadFile) -> Path:
    return await save_screenshot(image, settings.upload_path)


def _remove_upload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
