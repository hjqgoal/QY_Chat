from chat_reply_agent.services.quality_service import (
    _local_naturalness, review_candidates, similar_candidate_pairs,
)
from chat_reply_agent.services.reply_service import get_intent_conflict_issue


def review_targeted_replies(
    chat_text, replies, analysis, scenario, relation_rules,
    memory_summary, style_summary, controls, *, uncertain_speakers=False, candidate_indexes=None,
    style_evidence_available=None,
):
    targets = {}
    similar_indexes = {index for pair in similar_candidate_pairs(replies) for index in pair}
    for index, reply in enumerate(replies):
        if candidate_indexes is not None and index not in candidate_indexes:
            continue
        if reply.source == "fallback":
            continue
        reasons = []
        if analysis.risk_level != "low" or scenario.type in {"conflict", "boundary", "apology"}:
            reasons.append("检查冲突或关系边界，避免新增承诺、施压或激化矛盾")
        if scenario.confidence < 0.65 or uncertain_speakers:
            reasons.append("上下文或说话人证据不足，不要把猜测当成事实")
        conflict = get_intent_conflict_issue(reply.text, controls)
        if conflict:
            reasons.append(conflict)
        _, local_issues = _local_naturalness(reply.text, controls)
        # Broad expressions such as '如果你愿意' are not defects by themselves.
        if any(phrase in reply.text for phrase in (
            "我理解你的感受", "我们可以慢慢沟通", "有效沟通", "情绪价值",
        )):
            reasons.append("疑似咨询或客服腔，结合语境确认，必要时改为具体接话")
        if "超过本次设定的长度" in local_issues:
            reasons.append("检查能否在不丢失立场和关键信息的前提下缩短")
        if index in similar_indexes:
            reasons.append("候选措辞高度相近，检查是否只是换序；不能为了差异改变用户立场")
        if reasons:
            targets[index] = reasons

    return review_candidates(chat_text, replies, relation_rules, memory_summary,
                             style_summary, controls, targets,
                             style_evidence_available=style_evidence_available)
