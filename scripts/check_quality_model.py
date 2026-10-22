"""Opt-in, single-call provider check with synthetic text and no database access."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="用合成短对话检查一次真实评审调用；会产生模型费用，不读写个人记忆。")
    parser.add_argument("--live", action="store_true", help="明确允许调用当前配置的评审模型一次")
    args = parser.parse_args()
    if not args.live:
        parser.error("需要显式指定 --live；本次未调用模型。")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from chat_reply_agent.config import get_settings
    from chat_reply_agent.schemas import ReplyCandidate, ReplyControls
    from chat_reply_agent.services.llm_client import ModelCallError, configured_models, model_request_timeout, quality_model_limits
    from chat_reply_agent.services.quality_service import review_candidates
    from chat_reply_agent.services.skill_loader import load_relation_skill
    from chat_reply_agent.services.telemetry_service import trace_scope

    settings = get_settings()
    if not settings.openai_api_key:
        print(json.dumps({"status": "not_configured"}))
        return 1
    texts = ["好，你先忙", "嗯嗯，晚点聊", "行，那先这样"]
    with trace_scope() as trace, model_request_timeout(settings.quality_timeout_seconds), quality_model_limits():
        try:
            result = review_candidates("合成测试对话。对方：我去开个会，晚点再聊。", [
                ReplyCandidate(text=text, style="", reason="") for text in texts],
                load_relation_skill("friend"), "", "", ReplyControls(),
                {i: ["核对原文、关系分寸和候选差异"] for i in range(3)},
                audit_only=True, include_scores=False, raise_errors=True)
            valid = all(item.status == "reviewed" for item in result.report.items)
            outcome = {"status": "completed" if valid else "invalid_response",
                "reviewed_candidates": sum(item.status == "reviewed" for item in result.report.items),
                "verdicts": [item.verdict for item in result.report.items]}
        except ModelCallError as exc:
            valid = False
            outcome = {"status": "failed", "error_code": exc.kind}
        print(json.dumps({**outcome, "model": configured_models()["review"], **trace.snapshot()}, ensure_ascii=False, indent=2))
        return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
