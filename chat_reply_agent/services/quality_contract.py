"""Read-only wire protocol. The server, not the judge, owns original texts."""

from collections import Counter
from uuid import uuid4


DIMENSIONS = ("naturalness", "relevance", "relation_fit", "scenario_fit", "style_fit")


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


class ReadOnlyReviewContract:
    def __init__(self, originals: list[str], targets: dict):
        self.originals = tuple(originals)
        self.batch_id = uuid4().hex[:16]
        self.ids = [f"{self.batch_id}:{index}" for index in range(len(originals))]
        self.targets = {self.ids[index]: index for index in sorted(targets)}

    def candidates(self):
        return [{"candidate_id": key, "text": text}
                for key, text in zip(self.ids, self.originals)]

    def example(self):
        return {"items": [{"candidate_id": key, "review": {
                    **{name: None for name in DIMENSIONS},
                    "evidence": {name: "具体依据或缺失信息" for name in DIMENSIONS},
                    "issues": [], "hard_issues": [], "critical_unknowns": [], "advisories": [],
                }} for key in self.targets],
                "group_review": {"batch_id": self.batch_id, "distinct": None,
                                 "reason": "组内差异的具体依据"}}

    def schema(self):
        strings = {"type": "array", "maxItems": 3, "items": {"type": "string", "minLength": 1}}
        return _object({"items": {"type": "array", "minItems": len(self.targets), "maxItems": len(self.targets),
            "items": _object({"candidate_id": {"type": "string", "enum": list(self.targets)},
                "review": _object({
                    **{name: {"type": ["integer", "null"], "minimum": 1, "maximum": 5} for name in DIMENSIONS},
                    "evidence": _object({name: {"type": "string", "minLength": 1} for name in DIMENSIONS}),
                    **{name: strings for name in ("issues", "hard_issues", "critical_unknowns", "advisories")},
                })})},
            "group_review": _object({"batch_id": {"type": "string", "enum": [self.batch_id]},
                "distinct": {"type": ["boolean", "null"]}, "reason": {"type": "string", "minLength": 1}})})

    def bind(self, data: dict):
        """Return only uniquely identified rows; never infer an ID from order."""
        errors, bound = [], []
        def add(path, kind):
            errors.append({"path": path, "type": kind})
        rows = data.get("items")
        if not isinstance(rows, list):
            add("items", "missing_or_invalid_list")
            rows = []
        if len(rows) != len(self.targets):
            add("items", "wrong_item_count")
        ids = Counter(row.get("candidate_id") for row in rows
                      if isinstance(row, dict) and isinstance(row.get("candidate_id"), str))
        for position, row in enumerate(rows):
            path = f"items.{position}"
            key = row.get("candidate_id") if isinstance(row, dict) else None
            if not isinstance(key, str) or key not in self.targets:
                add(path + ".candidate_id", "unexpected_candidate_id")
                continue
            if ids[key] != 1:
                add(path + ".candidate_id", "duplicate_candidate_id")
                continue
            review = row.get("review")
            if not isinstance(review, dict):
                add(path + ".review", "missing_or_invalid_object")
                continue
            # Do not silently accept attempted rewrites or legacy text bindings.
            if set(row) != {"candidate_id", "review"} or set(review) - {
                *DIMENSIONS, "evidence", "issues", "hard_issues", "critical_unknowns", "advisories"
            }:
                add(path, "unexpected_fields")
                continue
            invalid_lists = [name for name in ("critical_unknowns", "advisories")
                             if not isinstance(review.get(name), list)
                             or any(not isinstance(value, str) or not value.strip() for value in review[name])]
            if invalid_lists:
                for name in invalid_lists:
                    add(path + ".review." + name, "missing_or_invalid_list")
                continue
            index = self.targets[key]
            bound.append((position, {"candidate_index": index,
                "reply": {"text": self.originals[index]},
                "review": {**review, "uncertainties": review["critical_unknowns"],
                           "evaluated_text": self.originals[index]}}))
        for key in self.targets:
            if key not in ids:
                add(f"candidates.{self.targets[key]}", "missing_target")
        group = data.get("group_review")
        if not isinstance(group, dict) or group.get("batch_id") != self.batch_id:
            add("group_review.batch_id", "batch_binding_mismatch")
            group = None
        elif (set(group) != {"batch_id", "distinct", "reason"}
              or not (group["distinct"] is None or type(group["distinct"]) is bool)
              or not isinstance(group["reason"], str) or not group["reason"].strip()):
            add("group_review", "missing_or_invalid_group_evidence")
            group = None
        else:
            group = {**group, "evaluated_texts": list(self.originals)}
        return bound, group, errors
