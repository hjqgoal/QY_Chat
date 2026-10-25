"""Offline rule replay only: no model calls, database access or semantic grades."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import yaml

from chat_reply_agent.schemas import ChatScenario, ReplyControls
from chat_reply_agent.services.quality_service import _local_naturalness
from chat_reply_agent.services.reply_service import normalize_reply_candidates


def compare():
    path = ROOT / "chat_reply_agent/evals/reply_policy_cases.yaml"
    dataset = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = []
    for case in dataset["cases"]:
        for probe in case["probes"]:
            text = probe["text"]
            score, issues = _local_naturalness(text, ReplyControls(length_preference="short"))
            normalized = normalize_reply_candidates([{"text": text}], {}, ChatScenario())[0]
            rows.append({"case": case["id"], "label": probe["label"],
                "old_local_naturalness": probe["old_local_naturalness"], "new_local_naturalness": score,
                "new_local_issues": issues, "text_and_paragraphs_preserved": normalized.text == text,
                "passed": score is None and not issues and normalized.text == text})
    return {"scope": dataset["purpose"], "baseline": "旧版默认短句上限22字的本地规则结果，非模型评分",
            "meaning": "好坏表达都不凭格式猜分；语义是否合适仍需独立评审与人工对照。",
            "passed": all(row["passed"] for row in rows), "cases": rows}


if __name__ == "__main__":
    report = compare()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
