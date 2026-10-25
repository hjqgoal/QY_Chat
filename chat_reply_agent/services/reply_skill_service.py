"""Read-only, allowlisted strategy resources. No model, network or database calls."""

from pathlib import Path

import yaml

from chat_reply_agent.services.telemetry_service import current_trace


SKILL_ROOT = Path(__file__).resolve().parent.parent / "skills" / "reply-strategies"
SKILL_IDS = ("clarify-facts", "assert-position", "take-responsibility", "pause-conflict", "respond-to-feelings")
SKILL_TOOL_POLICY = """
可选策略目录不是必须执行的工作流。普通接话可直接生成；需要具体方法时再调用load_reply_skills。
一次读取最多两个相关策略，整轮最多调用一次。可组合，但不得改变用户立场或覆盖基础约束。
仅凭关系类型或“冲突”标签不能默认选道歉。选择依据应来自本轮明确需求；不明确时保留不确定性。
读取失败或策略不适用时回到基础约束，不套用道歉/安抚模板，不假装成功读取。
策略是方法参考，不是事实证据，不能把文件里的例子当成用户经历或直接照搬。
工具参数只能包含目录中的skill_ids，禁止传聊天原文、身份、路径、URL或命令。
"""


def _read_skill(skill_id: str) -> tuple[dict, str]:
    if skill_id not in SKILL_IDS:
        raise ValueError("Unknown skill")
    path = (SKILL_ROOT / skill_id / "SKILL.md").resolve()
    if not path.is_relative_to(SKILL_ROOT.resolve()):
        raise ValueError("Invalid skill path")
    if path.stat().st_size > 16000:
        raise ValueError("Oversized skill")
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError("Missing metadata")
    meta = yaml.safe_load(parts[1])
    metadata = meta.get("metadata") if isinstance(meta, dict) else None
    if (not isinstance(meta, dict) or meta.get("name") != skill_id
            or not isinstance(meta.get("description"), str) or not meta["description"].strip()
            or len(meta["description"]) > 1024 or not isinstance(metadata, dict)
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items())
            or not metadata.get("version", "").strip() or not parts[2].strip()):
        raise ValueError("Invalid skill")
    return meta, parts[2].strip()


def reply_skill_catalog() -> list[dict]:
    catalog = []
    for skill_id in SKILL_IDS:
        try:
            meta, _ = _read_skill(skill_id)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        catalog.append({"id": skill_id, "description": meta["description"]})
    return catalog


def reply_skill_tool_definition() -> dict:
    return {"type": "function", "function": {"name": "load_reply_skills",
        "description": "按需读取聊天回复策略，一次选1-2个；日常接话可跳过，不执行策略或额外调用模型。",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["skill_ids"],
            "properties": {"skill_ids": {"type": "array", "minItems": 1, "maxItems": 2,
                "items": {"type": "string", "enum": list(SKILL_IDS)}}}}}}


def load_reply_skills(arguments: object) -> dict:
    valid = isinstance(arguments, dict) and set(arguments) == {"skill_ids"}
    ids = arguments.get("skill_ids") if valid else None
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 2
            or not all(isinstance(value, str) and value in SKILL_IDS for value in ids)
            or len(set(ids)) != len(ids)):
        return {"status": "invalid_arguments", "message": "只能选择1-2个不重复的目录ID；不接收路径或聊天。"}
    skills = []
    try:
        for skill_id in ids:
            meta, instructions = _read_skill(skill_id)
            skills.append({"id": skill_id, "version": meta["metadata"]["version"], "instructions": instructions})
    except (OSError, ValueError, yaml.YAMLError):
        return {"status": "unavailable", "message": "策略资料不可用，使用基础约束，不假装已读取。"}
    trace = current_trace.get()
    if trace:
        trace.reply_skills.extend({"id": item["id"], "version": item["version"]} for item in skills)
    return {"status": "ok", "skills": skills}
