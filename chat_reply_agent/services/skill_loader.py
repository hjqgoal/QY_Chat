from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
RELATION_SKILL_PACKAGES = {
    "friend": "friend-reply",
    "colleague": "colleague-reply",
    "family": "family-reply",
    "lover": "lover-committed-reply",
    "lover_committed": "lover-committed-reply",
    "lover_pursuing": "lover-pursuing-reply",
}


@lru_cache
def load_relation_skill(relation_type: str) -> dict[str, Any]:
    package_name = RELATION_SKILL_PACKAGES.get(relation_type, "friend-reply")
    package_dir = SKILLS_DIR / package_name
    rules_path = package_dir / "references" / "rules.yaml"

    if not rules_path.exists():
        package_dir = SKILLS_DIR / "friend-reply"
        rules_path = package_dir / "references" / "rules.yaml"

    data = _read_yaml(rules_path)
    skill_metadata = _read_skill_frontmatter(package_dir / "SKILL.md")
    if skill_metadata:
        data["skill"] = skill_metadata
    skill_text = (package_dir / "SKILL.md").read_text(encoding="utf-8")
    data["instructions"] = skill_text.split("---", 2)[2].strip() if skill_text.startswith("---") else skill_text.strip()
    return data if isinstance(data, dict) else {}


@lru_cache
def load_quality_skill() -> str:
    package = SKILLS_DIR / "reply-quality-review"
    instructions = (package / "SKILL.md").read_text(encoding="utf-8")
    # Load the body, not just the discovery metadata, into every review request.
    body = instructions.split("---", 2)[2].strip()
    rubric = (package / "references" / "rubric.md").read_text(encoding="utf-8")
    return body + "\n\n" + rubric


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def _read_skill_frontmatter(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    data = yaml.safe_load(parts[1]) or {}
    return data if isinstance(data, dict) else {}
