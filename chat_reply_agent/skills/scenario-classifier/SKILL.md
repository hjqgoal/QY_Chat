---
name: scenario-classifier
description: Classify Chinese WeChat conversation scenarios before reply generation. Use after screenshot OCR/extraction to infer the main scene, emotional risk, reply goal, cautions, timing, and natural examples for emotionally intelligent candidate replies.
---

# 聊天场景识别

读取 [场景规则](references/scenarios.yaml)，判断当前聊天最主要的场景。

## 使用方式

1. 只根据聊天证据判断，不脑补长期关系。
2. 优先看对方最后一条消息和最近几轮上下文。
3. 只选一个主要场景；证据不足时选 `daily_chat` 或 `unknown`。
4. 场景输出要服务回复生成：告诉回复该完成什么、避开什么、什么时候回更合适。

## 不要这样判断

- 不要只看某个关键词就定性。
- 不要把普通闲聊说成关系危机。
- 不要把用户或对方的临时情绪写成长期事实。
