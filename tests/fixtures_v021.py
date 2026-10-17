import json

GENERATION = json.dumps({
    "analysis": {"tone": "自然", "emotion": "平静", "attitude": "友好", "scene": "日常聊天",
                 "risk_level": "low", "key_points": ["回应当前话题"]},
    "scenario": {"type": "daily_chat", "confidence": 0.95, "signals": ["没有冲突信号"]},
    "strategy": "接住眼前这句话，不急着替对方安排。",
    "replies": [{"text": text, "style": style, "reason": "回应当前话题"} for text, style in
                [("那你先试试", "自然"), ("行，试完跟我说", "轻松"), ("看看能不能一起玩", "直接")]],
}, ensure_ascii=False)

MEMORY = json.dumps({"summary": "聊了周末火锅和最近的安排", "facts": [], "interaction_preferences": []}, ensure_ascii=False)
