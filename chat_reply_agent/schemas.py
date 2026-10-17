from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    speaker: str = Field(default="unknown")
    text: str
    time: str | None = None
    side: str | None = None
    bubble_center_x: float | None = Field(default=None, ge=0.0, le=1.0)
    text_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    speaker_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    speaker_source: str = Field(default="unknown")
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class ExtractedChat(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    topic: str = ""
    last_message_from_other: str = ""
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)
    review_required: bool = False
    review_message_indexes: list[int] = Field(default_factory=list)
    source_image_count: int = Field(default=1, ge=0, le=6)
    merged_overlap_count: int = Field(default=0, ge=0)

    def to_text(self) -> str:
        lines: list[str] = []
        for message in self.messages:
            speaker = {
                "me": "我",
                "other": "对方",
                "unknown": "未知",
            }.get(message.speaker, message.speaker)
            time = f" [{message.time}]" if message.time else ""
            side = f" ({message.side})" if message.side else ""
            lines.append(f"{speaker}{side}{time}: {message.text}")

        if self.topic:
            lines.append(f"对话主题：{self.topic}")
        if self.last_message_from_other:
            lines.append(f"对方最后一条消息：{self.last_message_from_other}")
        if self.notes:
            lines.append(f"识别备注：{'；'.join(self.notes)}")

        return "\n".join(lines)


class ChatAnalysis(BaseModel):
    tone: str = Field(default="未知")
    emotion: str = Field(default="未知")
    attitude: str = Field(default="未知")
    scene: str = Field(default="日常聊天")
    risk_level: str = Field(default="low")
    key_points: list[str] = Field(default_factory=list)


class ChatScenario(BaseModel):
    type: str = Field(default="daily_chat")
    label: str = Field(default="日常聊天")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    reply_goal: str = Field(default="自然回应对方当前话题")
    cautions: list[str] = Field(default_factory=list)


class TimingAdvice(BaseModel):
    should_reply_now: bool = True
    suggested_delay: str = "现在回复"
    reason: str = "当前没有明显冲突信号，可以自然回复。"


class ReplyCandidate(BaseModel):
    text: str
    style: str
    reason: str
    source: Literal["model", "fallback"] = "model"
    warnings: list[str] = Field(default_factory=list)


class ReplyControls(BaseModel):
    intent: Literal[
        "auto",
        "agree",
        "decline",
        "comfort",
        "explain",
        "advance",
        "cool_down",
        "end_topic",
        "ask_clarify",
    ] = "auto"
    custom_intent: str = Field(default="", max_length=160)
    length_preference: Literal["very_short", "short", "normal"] = "short"
    intimacy_level: int = Field(default=2, ge=1, le=3)
    initiative_level: int = Field(default=2, ge=1, le=3)


class ReplyQualityItem(BaseModel):
    candidate_index: int
    status: Literal["reviewed", "unavailable", "invalid_response", "not_reviewed"] = "not_reviewed"
    overall_score: int | None = Field(default=None, ge=0, le=100)
    naturalness: int | None = Field(default=None, ge=1, le=5)
    relevance: int | None = Field(default=None, ge=1, le=5)
    relation_fit: int | None = Field(default=None, ge=1, le=5)
    scenario_fit: int | None = Field(default=None, ge=1, le=5)
    style_fit: int | None = Field(default=None, ge=1, le=5)
    local_naturalness: int | None = Field(default=None, ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    revised: bool = False
    passed: bool = False


class ReplyQualityReport(BaseModel):
    review_targets: dict[int, list[str]] = Field(default_factory=dict)
    passed: bool = False
    status: Literal["reviewed", "unavailable", "invalid_response", "not_reviewed", "partial"] = "not_reviewed"
    average_score: int | None = Field(default=None, ge=0, le=100)
    summary: str = ""
    items: list[ReplyQualityItem] = Field(default_factory=list)


class QualityGateResult(BaseModel):
    replies: list[ReplyCandidate] = Field(default_factory=list)
    report: ReplyQualityReport


class ContactMemorySnapshot(BaseModel):
    contact_id: str = "default-contact"
    contact_name: str = "默认联系人"
    relation_type: str = "friend"
    summary: str = ""
    facts: list[str] = Field(default_factory=list)
    interaction_preferences: list[str] = Field(default_factory=list)
    last_topic: str = ""
    last_scenario: str = ""
    has_previous_memory: bool = False


class ContactListResponse(BaseModel):
    contacts: list[ContactMemorySnapshot] = Field(default_factory=list)


MemoryFact = Annotated[str, Field(min_length=1, max_length=80)]


class ContactMemoryUpdateRequest(BaseModel):
    user_id: str = Field(default="default", min_length=1, max_length=64)
    contact_name: str = Field(default="", max_length=64)
    relation_type: Literal[
        "lover_committed",
        "lover_pursuing",
        "colleague",
        "family",
        "friend",
    ] = "friend"
    summary: str = Field(default="", max_length=240)
    facts: list[MemoryFact] = Field(default_factory=list, max_length=8)
    interaction_preferences: list[MemoryFact] = Field(
        default_factory=list,
        max_length=5,
    )
    last_topic: str = Field(default="", max_length=120)
    last_scenario: str = Field(default="", max_length=64)


class ContactMemoryDeleteResponse(BaseModel):
    contact_id: str
    deleted_contact_memory: int = 0
    deleted_memory_episodes: int = 0
    deleted_conversation_memories: int = 0
    cancelled_memory_jobs: int = 0

    @property
    def deleted(self) -> bool:
        return any(
            (
                self.deleted_contact_memory,
                self.deleted_memory_episodes,
                self.deleted_conversation_memories,
                self.cancelled_memory_jobs,
            )
        )


class RetrievedMemoryItem(BaseModel):
    id: int
    topic: str = ""
    scenario_type: str = ""
    scenario_label: str = ""
    summary: str = ""
    relevance_score: int = Field(default=0, ge=0, le=100)
    similarity_score: float | None = Field(default=None, ge=0, le=1)
    time_decay_factor: float | None = Field(default=None, ge=0, le=1)
    decay_half_life_days: float | None = Field(default=None, gt=0)
    similarity_method: Literal[
        "semantic", "lexical", "lexical_fallback", "not_scored"
    ] = "not_scored"
    created_at: datetime


class MemorySearchResponse(BaseModel):
    contact_id: str
    query: str = ""
    memories: list[RetrievedMemoryItem] = Field(default_factory=list)


class MemoryEpisodeDeleteResponse(BaseModel):
    contact_id: str
    memory_id: int
    deleted: bool = True


class ReplyContextInput(BaseModel):
    background_context: str = Field(default="", max_length=2000)

    @field_validator("background_context", mode="before")
    @classmethod
    def normalize_background_context(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


class CorrectedReplyRequest(ReplyContextInput):
    relation_type: str
    user_id: str = "default"
    contact_id: str = "default-contact"
    contact_name: str = ""
    extracted_chat: ExtractedChat
    controls: ReplyControls = Field(default_factory=ReplyControls)
    save_to_memory: bool = True


class PreparedReplyRequest(ReplyContextInput):
    relation_type: str
    user_id: str = "default"
    contact_id: str = "default-contact"
    contact_name: str = ""
    extracted_chat: ExtractedChat
    controls: ReplyControls = Field(default_factory=ReplyControls)
    save_to_memory: bool = True


class RegenerateReplyRequest(ReplyContextInput):
    relation_type: str
    user_id: str = "default"
    contact_id: str = "default-contact"
    contact_name: str = ""
    extracted_chat: ExtractedChat
    analysis: ChatAnalysis
    scenario: ChatScenario
    existing_replies: list[ReplyCandidate] = Field(min_length=3, max_length=3)
    candidate_index: int = Field(ge=0, le=2)
    controls: ReplyControls = Field(default_factory=ReplyControls)


class RegenerateReplyResponse(BaseModel):
    weather_evidence: list[dict] = Field(default_factory=list)
    candidate_index: int
    reply: ReplyCandidate
    quality: ReplyQualityItem
    diagnostics: dict = Field(default_factory=dict)


class ReplyFeedbackRequest(BaseModel):
    relation_type: str
    user_id: str = "default"
    feedback_type: str
    candidate_index: int
    candidate_text: str
    edited_text: str | None = None
    memory_id: int | None = None


class ReplyFeedbackResponse(BaseModel):
    feedback_id: int
    preference_summary: str
    style_examples_summary: str = ""


class StyleExampleItem(BaseModel):
    id: int
    text: str
    source: str
    created_at: datetime | None = None


class StyleExampleCreateRequest(BaseModel):
    user_id: str = Field(default="default", min_length=1, max_length=64)
    relation_type: Literal[
        "lover_committed",
        "lover_pursuing",
        "colleague",
        "family",
        "friend",
    ]
    text: str = Field(min_length=1, max_length=80)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = value.replace("\n", " ").strip()
        if not cleaned:
            raise ValueError("风格样本不能为空。")
        return cleaned


class StyleExampleDeleteResponse(BaseModel):
    example_id: int
    deleted: bool = True


class StyleResetResponse(BaseModel):
    relation_type: str
    deleted_profiles: int = 0
    deleted_examples: int = 0
    deleted_feedback: int = 0


class StyleExamplesResponse(BaseModel):
    examples: list[StyleExampleItem]
    summary: str
    automatic_summary: str = ""
    automatic_examples: list[str] = Field(default_factory=list)


class StyleEvidence(BaseModel):
    current_user_message_count: int = Field(default=0, ge=0)
    current_summary: str = ""
    current_examples: list[str] = Field(default_factory=list)
    profile_summary: str = ""
    profile_examples: list[str] = Field(default_factory=list)


class ReplyResponse(BaseModel):
    weather_evidence: list[dict] = Field(default_factory=list)
    user_id: str = "default"
    relation_type: str
    controls: ReplyControls = Field(default_factory=ReplyControls)
    memory_persisted: bool = True
    memory_job_id: str | None = None
    memory_status: Literal["disabled", "pending", "running", "completed", "failed", "cancelled"] = "disabled"
    contact_id: str = "default-contact"
    contact_name: str = "默认联系人"
    extracted_chat: ExtractedChat
    background_context: str = ""
    screenshot_text: str
    analysis: ChatAnalysis
    scenario: ChatScenario
    timing: TimingAdvice
    strategy: str
    replies: list[ReplyCandidate]
    quality: ReplyQualityReport
    memory_summary: str
    style_summary: str
    style_evidence: StyleEvidence = Field(default_factory=StyleEvidence)
    preference_summary: str = ""
    style_examples_summary: str = ""
    retrieved_memories: list[RetrievedMemoryItem] = Field(default_factory=list)
    contact_memory: ContactMemorySnapshot
    saved_memory_id: int | None = None
    saved_correction_id: int | None = None
    diagnostics: dict = Field(default_factory=dict)


class EvalExpectation(BaseModel):
    scenario_types: list[str] = Field(default_factory=list)
    max_reply_chars: int = Field(default=32, ge=4, le=200)
    forbidden_phrases: list[str] = Field(default_factory=list)
    reference_replies: list[str] = Field(default_factory=list)
    notes: str = ""


class EvalCase(BaseModel):
    id: str
    name: str
    relation_type: str
    messages: list[ChatMessage] = Field(default_factory=list)
    image_path: str | None = None
    expected: EvalExpectation = Field(default_factory=EvalExpectation)

    def expected_chat(self) -> ExtractedChat:
        last_other = ""
        for message in self.messages:
            if message.speaker == "other":
                last_other = message.text
        return ExtractedChat(
            messages=self.messages,
            last_message_from_other=last_other,
            confidence=1.0,
            notes=["评测集人工标注内容"],
        )


class EvalCaseSummary(BaseModel):
    id: str
    name: str
    relation_type: str
    expected_scenarios: list[str] = Field(default_factory=list)
    has_image: bool = False
    message_count: int = 0


class EvalCaseListResponse(BaseModel):
    dataset_name: str
    version: str
    cases: list[EvalCaseSummary] = Field(default_factory=list)


class EvalRunRequest(BaseModel):
    case_ids: list[str] = Field(min_length=1, max_length=20)
    dataset: Literal["core", "v020"] = "core"


class EvalMetrics(BaseModel):
    scenario_accuracy: int = Field(default=0, ge=0, le=100)
    reply_quality: int | None = Field(default=None, ge=0, le=100)
    constraint_compliance: int = Field(default=0, ge=0, le=100)
    candidate_diversity: int = Field(default=0, ge=0, le=100)
    recognition_accuracy: int | None = Field(default=None, ge=0, le=100)
    text_accuracy: int | None = Field(default=None, ge=0, le=100)
    speaker_accuracy: int | None = Field(default=None, ge=0, le=100)


class EvalCaseResult(BaseModel):
    case_id: str
    name: str
    relation_type: str
    passed: bool = False
    score: int | None = Field(default=None, ge=0, le=100)
    expected_scenarios: list[str] = Field(default_factory=list)
    actual_scenario: str = ""
    metrics: EvalMetrics = Field(default_factory=EvalMetrics)
    replies: list[ReplyCandidate] = Field(default_factory=list)
    reference_replies: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    error: str | None = None


class EvalRunResponse(BaseModel):
    dataset_name: str
    version: str
    passed: bool = False
    average_score: int | None = Field(default=None, ge=0, le=100)
    passed_cases: int = 0
    total_cases: int = 0
    results: list[EvalCaseResult] = Field(default_factory=list)
