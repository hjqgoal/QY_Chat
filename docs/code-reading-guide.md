# 面试版代码阅读路线

对应 2026-10-23 验收的本地 v0.22。先弄清当前主流程，不必按文件名从头读完所有历史服务。

## 第一步：跑通一次请求

1. `chat_reply_agent/static/index.html`：表单、截图预览/校对、生成提交、异步评估和记忆状态轮询。
2. `chat_reply_agent/schemas.py`：请求、聊天消息、回复、检索诊断、后台任务的结构。留意哪些状态表示“已完成”，哪些只是“尚未验证”。
3. `chat_reply_agent/main.py`：重点读 `/api/vision/extract-batch`、`/api/reply/from-chat`、`/api/reply/corrected`、`_build_reply_response`、`_load_generation_context`。

主线：

```text
截图 -> 文件校验 -> 视觉提取 -> 用户校对
文字背景 --------------------------+
                                   v
                历史召回 + 关系 Skill + 本人风格
                                   v
                     强模型理解并生成
                     (必要时天气 MCP 往返)
                                   v
                    主题 / 时机 / 策略 / 三条候选
                       |                       |
               后台质量评估                后台记忆任务
               一次修订和复评              SQLite 保存
```

要能解释：为什么保留人工校对；为什么概括性背景不是逐条聊天原话；为什么首个响应返回时记忆可能仍未保存。

## 第二步：理解模型边界

- `services/vision_service.py`：提取格式、从左右位置计算说话人、重复消息与跨截图重叠、未知归属及复核标记。当前位置也来自模型，不能当作可靠检测器的实测坐标。
- `services/upload_service.py`：真实图片解码、类型/大小限制和清理。
- `services/generation_service.py`：当前主入口 `understand_and_generate`，强模型一次输出主题、分析、场景、策略和候选，再做结构校验。
- `services/llm_client.py`：配置的模型角色、请求超时、结构化输出、错误脱敏、受限天气 tool-call 往返。
- `services/skill_loader.py` 与 `skills/`：读取关系 `SKILL.md` 正文和 YAML 规则，以及独立质量评审准则。它们影响提示，不是微调权重。
- `services/timing_service.py`：把分析/场景转成回复时机建议。

不要仅因为目录中存在 `analysis_service.py`、`reply_service.py` 等文件，就把旧版逐步调用流程讲成当前主路径；以 `main.py` 的实际调用为准。

## 第三步：数据库与 RAG

- `config.py`、`database.py`：`.env` 配置、缓存、数据库连接及会话管理。配置修改后重启。
- `models.py`：`ContactMemory` 是联系人概要/事实；`MemoryEpisode` 是供检索的历史片段；`ConversationMemory` 是一轮保存内容；`UserStyleProfile`、`StyleExample`、`ReplyFeedback` 分别处理自动风格、示例与反馈；`MemoryJob` 保存后台任务；`ChatCorrection` 保存校对记录。
- `services/retrieval_service.py`：先按 user_id/contact_id 选候选，再计算相似度与时间衰减，最后拼接上下文。
- `services/embedding_service.py`：当前千问原生向量接口、独立向量分批、顺序与数值校验、余弦相似度、严格失败策略。
- `services/contact_memory_service.py`、`memory_service.py`、`style_service.py`：摘要、事实、本人风格各自负责什么。只有确认后的本人逐条消息参与自动风格学习，不能从对方或概括性背景学习本人语气。
- `services/data_management_service.py`：编辑、删除与后台迟到结果之间的约束。

公式：`相关性 = 语义余弦相似度 × 2 ** (-年龄天数 / 半衰期天数)`。

要能解释：向量模型只编码文本，回复模型才生成回复；RAG 不等于向量数据库；当前最多 60 条候选、默认最多 3 条召回、向量不持久化；为什么更换模型后要重新评估召回阈值。

## 第四步：异步、工具与失败处理

- `services/memory_queue_service.py`：SQLite 持久任务、独立会话、单进程锁、启动恢复、原子写入、网络请求后重新检查取消。
- `services/quality_queue_service.py`：有界线程池、阶段/总预算、取消、过期和原稿保护。质量任务在内存，不具备记忆任务的持久性。
- `services/quality_service.py`、`quality_repair_service.py`：评价原文、针对问题修订、再评估；不自动循环、不把修订成功等同于质量合格。
- `services/mcp_service.py`、`mcp_server.py`：启动固定工具服务，通过 MCP 协议调用天气工具。它不是新增了一个有独立人格的聊天 Agent。
- `services/weather_contract.py`、`weather_service.py`：只允许地点/日期参数；城市消歧、当地时间、单位、数据缺失、AQI 标准和失败处理。
- `services/telemetry_service.py`：单次请求与后台阶段的耗时，理解为什么不能把一次耗时当 P95。

## 第五步：沿着测试验证你的理解

- 主流程：`tests/test_v021_pipeline.py`、`test_structured_generation.py`、`test_topic_context.py`。
- 图片与输入：`test_vision_service.py`、`test_demo_readiness.py`、`test_context_service.py`。
- 语义记忆：`test_retrieval_service.py`、`test_semantic_rag.py`、`test_qwen_embedding.py`。
- 后台与质量：`test_quality_queue.py`、`test_quality_workflow.py`、`test_quality_rubric.py`。
- 数据与工具：`test_data_management.py`、`test_style_service.py`、`test_weather.py`。
- 页面状态：`test_frontend_quality_browser.cjs`，读用户修改后迟到评价不能覆盖文本的用例。

练习时选一个测试，先预测预期输出，再找实现证明它。能说明一个真实失败及修复过程，通常比背全部函数名更有说服力。

## 面试表述边界

准确表述：显式 Python 编排的本地 AI 回复助手，包含视觉识别与人工确认、关系规则、语义记忆检索、受限 MCP 工具和异步质量反馈。

不要表述成：LangGraph 项目、训练了专属模型、生产级多租户系统、全量 ANN 向量库、100% 情绪准确率、自动替用户发微信，或稳定秒回。

展示验收与已知风险见 [最终检查记录](release-acceptance-2026-10-23.md)。
