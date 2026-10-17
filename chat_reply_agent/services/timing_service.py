from chat_reply_agent.schemas import ChatAnalysis, ChatScenario, TimingAdvice
from chat_reply_agent.services.scenario_service import get_scenario_rule


def suggest_reply_timing(
    analysis: ChatAnalysis,
    scenario: ChatScenario,
    relation_rules: dict,
) -> TimingAdvice:
    scenario_timing = get_scenario_rule(scenario.type).get("timing", {})

    if analysis.risk_level == "high":
        return TimingAdvice(
            should_reply_now=False,
            suggested_delay="先冷静 20-40 分钟再回复",
            reason="当前风险较高，立即回复容易放大误解，先整理态度和重点更稳妥。",
        )

    if scenario_timing.get("should_reply_now") is False:
        return TimingAdvice(
            should_reply_now=False,
            suggested_delay=scenario_timing.get("default_delay", "稍等后再回复"),
            reason=scenario_timing.get("reason", "当前更适合先留一点回应空间。"),
        )

    if analysis.risk_level == "medium":
        return TimingAdvice(
            should_reply_now=True,
            suggested_delay="5-10 分钟内回复",
            reason="对方可能已有情绪，但还适合及时回应，语气要轻、先接住感受。",
        )

    relation_timing = relation_rules.get("timing", {})
    return TimingAdvice(
        should_reply_now=bool(scenario_timing.get("should_reply_now", True)),
        suggested_delay=scenario_timing.get(
            "default_delay",
            relation_timing.get("default_delay", "现在回复"),
        ),
        reason=scenario_timing.get(
            "reason",
            relation_timing.get("reason", "当前没有明显冲突信号，可以自然回复。"),
        ),
    )
