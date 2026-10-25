from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from chat_reply_agent.schemas import (
    ChatScenario,
    EvalCase,
    EvalCaseListResponse,
    EvalCaseResult,
    EvalCaseSummary,
    EvalMetrics,
    EvalRunResponse,
    ExtractedChat,
    ReplyCandidate,
    ReplyQualityReport,
)
from chat_reply_agent.services.generation_service import understand_and_generate
from chat_reply_agent.services.context_service import build_model_chat_context
from chat_reply_agent.services.directed_review_service import review_targeted_replies
from chat_reply_agent.schemas import ReplyControls
from chat_reply_agent.services.weather_contract import weather_evidence_context
from chat_reply_agent.services.llm_client import ModelCallError
from chat_reply_agent.services.quality_service import review_and_refine_replies
from chat_reply_agent.services.skill_loader import load_relation_skill
from chat_reply_agent.services.vision_service import extract_chat_from_image


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "evals" / "dataset.yaml"
)


def load_evaluation_dataset(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
) -> tuple[str, str, list[EvalCase]]:
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError("评测数据集必须是 YAML 对象。")

    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("评测数据集的 cases 必须是数组。")
    cases = [EvalCase.model_validate(item) for item in raw_cases]
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("评测样本 id 不能重复。")

    return (
        str(payload.get("name", "chat_reply_agent core evals")),
        str(payload.get("version", "1")),
        cases,
    )


def list_evaluation_cases(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
) -> EvalCaseListResponse:
    name, version, cases = load_evaluation_dataset(dataset_path)
    return EvalCaseListResponse(
        dataset_name=name,
        version=version,
        cases=[
            EvalCaseSummary(
                id=case.id,
                name=case.name,
                relation_type=case.relation_type,
                expected_scenarios=case.expected.scenario_types,
                has_image=bool(case.image_path),
                message_count=len(case.messages),
            )
            for case in cases
        ],
    )


def run_evaluation(
    case_ids: list[str],
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
) -> EvalRunResponse:
    dataset_name, version, cases = load_evaluation_dataset(dataset_path)
    case_map = {case.id: case for case in cases}
    unknown_ids = [case_id for case_id in case_ids if case_id not in case_map]
    if unknown_ids:
        raise ValueError(f"找不到评测样本：{', '.join(unknown_ids)}")

    selected = [case_map[case_id] for case_id in dict.fromkeys(case_ids)]
    results = [run_evaluation_case(case, Path(dataset_path)) for case in selected]
    average_score = (
        round(sum(result.score for result in results) / len(results))
        if results and all(result.score is not None for result in results) else None
    )
    passed_cases = sum(result.passed for result in results)
    return EvalRunResponse(
        dataset_name=dataset_name,
        version=version,
        passed=bool(results) and passed_cases == len(results),
        average_score=average_score,
        passed_cases=passed_cases,
        total_cases=len(results),
        results=results,
    )


def run_evaluation_case(case: EvalCase, dataset_path: Path) -> EvalCaseResult:
    expected_chat = case.expected_chat()
    try:
        actual_chat = _load_case_chat(case, dataset_path, expected_chat)
        relation_rules = load_relation_skill(case.relation_type)
        chat_text = build_model_chat_context(actual_chat, case.background_context)
        generated = understand_and_generate(
            chat_text=chat_text,
            relation_rules=relation_rules,
            memory_summary="",
            style_summary="",
        )
        analysis, scenario = generated.analysis, generated.scenario
        chat_text += weather_evidence_context(generated.weather_evidence)
        production = review_targeted_replies(chat_text, generated.replies, analysis, scenario,
            relation_rules, "", "", ReplyControls(), uncertain_speakers=actual_chat.review_required)
        replies = production.replies
        quality_result = review_and_refine_replies(
            chat_text=chat_text,
            replies=replies,
            analysis=analysis,
            scenario=scenario,
            relation_rules=relation_rules,
            memory_summary="",
            style_summary="",
            audit_only=True,
        )
        return score_evaluation_output(
            case=case,
            actual_chat=actual_chat,
            scenario=scenario,
            replies=quality_result.replies,
            quality=quality_result.report,
        )
    except (ModelCallError, OSError, ValueError) as exc:
        return EvalCaseResult(
            case_id=case.id,
            name=case.name,
            relation_type=case.relation_type,
            expected_scenarios=case.expected.scenario_types,
            reference_replies=case.expected.reference_replies,
            issues=["评测执行失败"],
            error=str(exc),
        )


def score_evaluation_output(
    case: EvalCase,
    actual_chat: ExtractedChat,
    scenario: ChatScenario,
    replies: list[ReplyCandidate],
    quality: ReplyQualityReport,
) -> EvalCaseResult:
    expected_scenarios = case.expected.scenario_types
    scenario_score = (
        100 if not expected_scenarios or scenario.type in expected_scenarios else 0
    )
    constraint_score, constraint_issues = _score_reply_constraints(case, replies)
    diversity_score = _score_candidate_diversity(replies)
    recognition = _score_recognition(case.expected_chat(), actual_chat)
    quality_score = quality.average_score

    if case.image_path:
        weights = {
            "recognition": 0.30,
            "scenario": 0.20,
            "quality": 0.25,
            "constraints": 0.15,
            "diversity": 0.10,
        }
        score = round(
            (recognition["overall"] or 0) * weights["recognition"]
            + scenario_score * weights["scenario"]
            + (quality_score or 0) * weights["quality"]
            + constraint_score * weights["constraints"]
            + diversity_score * weights["diversity"]
        )
    else:
        score = round(
            scenario_score * 0.25
            + (quality_score or 0) * 0.35
            + constraint_score * 0.25
            + diversity_score * 0.15
        )

    issues = list(constraint_issues) + list(quality.group_issues)
    for item in quality.items:
        issues.extend(f"候选 {item.candidate_index + 1}：{issue}" for issue in item.issues + item.uncertainties)
    if quality_score is None:
        score = None
        issues.append("质量评估不可计分或依据不足，本次不计算综合评分")
    if scenario_score == 0:
        issues.append(
            f"场景应为 {' / '.join(expected_scenarios)}，实际为 {scenario.type}"
        )
    if quality_score is not None and quality_score < 75:
        issues.append(f"回复质量均分只有 {quality_score}")
    if diversity_score < 60:
        issues.append("三条候选回复差异不够明显")
    if case.image_path and (recognition["text"] or 0) < 90:
        issues.append("截图文字识别准确率不足 90%")
    if case.image_path and (recognition["speaker"] or 0) < 90:
        issues.append("截图说话人识别准确率不足 90%")

    recognition_passed = not case.image_path or (recognition["overall"] or 0) >= 85
    has_hard_constraint_issue = any(
        "候选回复数量" in issue
        or ("候选" in issue and "为空" in issue)
        or "禁用表达" in issue
        for issue in constraint_issues
    )
    passed = (
        score is not None
        and score >= 75
        and quality.passed
        and scenario_score == 100
        and constraint_score >= 80
        and recognition_passed
        and not has_hard_constraint_issue
        and not any(reply.text.strip() == "" for reply in replies)
    )
    return EvalCaseResult(
        case_id=case.id,
        name=case.name,
        relation_type=case.relation_type,
        passed=passed,
        score=score,
        expected_scenarios=expected_scenarios,
        actual_scenario=scenario.type,
        metrics=EvalMetrics(
            scenario_accuracy=scenario_score,
            reply_quality=quality_score,
            constraint_compliance=constraint_score,
            candidate_diversity=diversity_score,
            recognition_accuracy=recognition["overall"] if case.image_path else None,
            text_accuracy=recognition["text"] if case.image_path else None,
            speaker_accuracy=recognition["speaker"] if case.image_path else None,
        ),
        replies=replies,
        reference_replies=case.expected.reference_replies,
        issues=list(dict.fromkeys(issues)),
    )


def _load_case_chat(
    case: EvalCase,
    dataset_path: Path,
    expected_chat: ExtractedChat,
) -> ExtractedChat:
    if not case.image_path:
        return expected_chat
    image_path = Path(case.image_path)
    if not image_path.is_absolute():
        image_path = dataset_path.resolve().parent / image_path
    if not image_path.is_file():
        raise ValueError(f"评测截图不存在：{image_path}")
    return extract_chat_from_image(image_path)


def _score_recognition(
    expected: ExtractedChat,
    actual: ExtractedChat,
) -> dict[str, int | None]:
    expected_messages = expected.messages
    actual_messages = actual.messages
    total = max(len(expected_messages), len(actual_messages))
    if total == 0:
        return {"overall": 100, "text": 100, "speaker": 100}

    text_points = 0.0
    speaker_points = 0
    for expected_item, actual_item in zip(expected_messages, actual_messages):
        text_points += SequenceMatcher(
            None,
            _normalize_text(expected_item.text),
            _normalize_text(actual_item.text),
        ).ratio()
        if expected_item.speaker == actual_item.speaker:
            speaker_points += 1

    text_score = round(text_points / total * 100)
    speaker_score = round(speaker_points / total * 100)
    overall = round(text_score * 0.7 + speaker_score * 0.3)
    return {"overall": overall, "text": text_score, "speaker": speaker_score}


def _score_reply_constraints(
    case: EvalCase,
    replies: list[ReplyCandidate],
) -> tuple[int, list[str]]:
    if len(replies) != 3:
        return 0, [f"候选回复数量应为 3，实际为 {len(replies)}"]

    forbidden = list(dict.fromkeys(case.expected.forbidden_phrases))
    total_checks = len(replies) * 3
    passed_checks = 0
    issues: list[str] = []
    for index, reply in enumerate(replies, start=1):
        text = reply.text.strip()
        if text:
            passed_checks += 1
        else:
            issues.append(f"候选 {index} 为空")
        if 2 <= len(text) <= case.expected.max_reply_chars:
            passed_checks += 1
        else:
            issues.append(
                f"候选 {index} 长度为 {len(text)}，上限为 {case.expected.max_reply_chars}"
            )
        matched = [phrase for phrase in forbidden if phrase and phrase in text]
        if not matched:
            passed_checks += 1
        else:
            issues.append(f"候选 {index} 包含禁用表达：{'、'.join(matched[:3])}")
    return round(passed_checks / total_checks * 100), issues


def _score_candidate_diversity(replies: list[ReplyCandidate]) -> int:
    if len(replies) < 2:
        return 0
    texts = [_normalize_text(reply.text) for reply in replies]
    similarities = [
        SequenceMatcher(None, left, right).ratio()
        for left, right in combinations(texts, 2)
    ]
    text_score = round(100 - (sum(similarities) / len(similarities) * 50))
    style_score = round(len({reply.style.strip() for reply in replies}) / len(replies) * 100)
    return min(100, max(0, round(text_score * 0.8 + style_score * 0.2)))


def _normalize_text(value: str) -> str:
    punctuation = " ，。！？、；：,.!?;:\n\t"
    return "".join(character for character in value.lower() if character not in punctuation)
