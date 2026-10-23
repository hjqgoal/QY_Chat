"""Synthetic judge responses for contract tests, not evidence of model quality."""

from copy import deepcopy
import json

DIMENSIONS = ("naturalness", "relevance", "relation_fit", "scenario_fit", "style_fit")


def reviewed_payload(texts, originals=None):
    originals = originals or texts
    return {
        "items": [{
            "candidate_index": index, "reply": {"text": text},
            "review": {
                "evaluated_text": originals[index], **{key: 4 for key in DIMENSIONS},
                "evidence": {key: f"以原文‘{originals[index]}’检查 {key}，无新增安排。" for key in DIMENSIONS},
                "issues": [], "hard_issues": [], "uncertainties": [],
            },
        } for index, text in enumerate(texts)],
        "group_review": {"evaluated_texts": originals, "distinct": True,
                         "reason": "候选分别表达尝试、条件和后续沟通。"},
    }


def judge_responses(*payloads):
    """Encode existing test judgments using the IDs of each actual request."""
    responses = iter(payloads)
    def respond(*args, **kwargs):
        payload = next(responses)
        if isinstance(payload, Exception):
            raise payload
        return judge_response(payload, *args, **kwargs)
    return respond


def judge_response(payload, *args, **kwargs):
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return payload
    schema = kwargs.get("response_schema", {})
    properties = schema.get("properties", {}).get("items", {}).get("items", {}).get("properties", {})
    if "candidate_id" not in properties or "items" not in payload:
        return json.dumps(payload)
    data = deepcopy(payload)
    context = json.loads(args[1].split("输入数据：\n", 1)[1].split("\n输出结构：", 1)[0])
    ids = [row["candidate_id"] for row in context["candidates"]]
    originals = [row["text"] for row in context["candidates"]]
    for row in data["items"]:
        index = row.pop("candidate_index")
        row["candidate_id"] = ids[index]
        reply = row.pop("reply")
        evaluated = row["review"].pop("evaluated_text")
        if "uncertainties" in row["review"]:
            row["review"]["critical_unknowns"] = row["review"].pop("uncertainties")
        row["review"].setdefault("advisories", [])
        if reply["text"] != originals[index] or evaluated != originals[index]:
            row["reply"] = reply  # Deliberate invalid response remains invalid.
    group = data.get("group_review", {})
    group["batch_id"] = context["batch_id"] if group.pop("evaluated_texts", None) == originals else "stale"
    data["group_review"] = group
    return json.dumps(data)
