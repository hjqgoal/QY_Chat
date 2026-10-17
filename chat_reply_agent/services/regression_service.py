"""Offline replay of the old replacement rule, not an LLM quality benchmark."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, StrictStr

from chat_reply_agent.schemas import ChatScenario
from chat_reply_agent.services.evaluation_service import load_evaluation_dataset
from chat_reply_agent.services.reply_service import (
    AI_SMELL_PHRASES, _fallback_replies, normalize_reply_candidates,
)
from chat_reply_agent.services.scenario_service import get_scenario_rule
from chat_reply_agent.services.skill_loader import load_relation_skill


DATASET_PATH = Path(__file__).resolve().parent.parent / "evals" / "v020_regressions.yaml"


class NormalizationCheck(BaseModel):
    id: StrictStr
    input: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    issue: str = Field(min_length=1)


def compare_reply_regressions() -> dict:
    payload = yaml.safe_load(DATASET_PATH.read_text(encoding="utf-8"))
    name, version, cases = load_evaluation_dataset(DATASET_PATH)
    checks = [NormalizationCheck.model_validate(item) for item in payload["normalization_checks"]]
    rules = load_relation_skill(cases[0].relation_type)
    scenario = ChatScenario(type="daily_chat", label="日常聊天")
    fallback = _fallback_replies(rules, get_scenario_rule(scenario.type))[0]["text"]
    results = []
    for check in checks:
        legacy_replaced = len(check.input) > 46 or any(phrase in check.input for phrase in AI_SMELL_PHRASES)
        result = normalize_reply_candidates([
            {"text": check.input, "style": "待检查", "reason": "人工构造的复现候选"},
        ], rules, scenario)[0]
        results.append({
            "id": check.id,
            "input": check.input,
            "before": fallback if legacy_replaced else check.input,
            "after": result.text,
            "reference": check.reference,
            "issue": check.issue,
            "legacy_replaced": legacy_replaced,
            "passed": result.text == check.input and result.source == "model",
        })
    return {
        "dataset_name": name, "version": version,
        "source": payload["source"],
        "mode": "offline_rule_replay",
        "model_called": False,
        "passed": all(item["passed"] for item in results),
        "chat": cases[0].expected_chat().model_dump(),
        "results": results,
    }
