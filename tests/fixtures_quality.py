"""Synthetic judge responses for contract tests, not evidence of model quality."""

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
