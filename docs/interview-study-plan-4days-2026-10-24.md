# 聊天助手项目：4 天 × 5 小时面试学习执行手册

适用目标：AI 应用开发 / 大模型应用工程岗位，重点准备项目架构与关键技术追问。

版本依据：2026-10-24 当前工作区，包括尚未提交的改动。文中的代码位置按本次核对版本填写；后续若行号移动，用函数名定位。真实模型选择和运行配置以实际加载结果为准，本手册没有读取个人密钥或聊天数据库。

本次复核已运行：290 项后端离线测试通过，4 组前端逻辑检查通过。没有重新进行真实模型调用、联网天气验收或浏览器回归。测试通过说明这些用例覆盖的工程行为符合断言，不代表真实模型的语义准确率。

## 使用方法与最终交付

每天恰好 300 分钟，拆成 6 个 50 分钟时段。每个时段建议前 40 分钟操作、接着 5 分钟记录、最后 5 分钟休息；休息包含在 5 小时内。时间表是相对开始时间，不要求固定几点开工。

所有“必做”实验均可以用现有离线测试或纯本地计算完成，不要求消耗真实模型费用。遇到环境故障，最多排查 15 分钟；仍未解决就先阅读该测试的输入、替身和断言，记录“未执行”，继续学习，最后再集中处理。

创建一份自己的笔记，每个机制固定写六项：输入、输出、调用者、失败分支、设计取舍、证明它的测试。先预测结果，再运行测试，最后解释结果；不要只记录“运行成功”。

四天结束应交付：

- [ ] 一张当前架构图和一张完整请求时序图。
- [ ] 一张核心数据表与隔离维度图。
- [ ] 一份 RAG 手算结果和失败策略比较。
- [ ] 两张后台任务状态图，说明持久性与取消语义。
- [ ] 三个工程故事：识图恢复、记忆一致性、评审协议与分级。
- [ ] 90 秒与 3 分钟项目介绍，以及一页扩展方案。
- [ ] 对文末 20 道追问逐题录音自测。

## 先更新对项目的认识

| 主题 | 当前实现 | 学习时要纠正的旧印象 |
| --- | --- | --- |
| 图片上传 | 流式限量写入，Pillow 验证与解码；单张 20 MiB、整批 60 MiB、单图 4000 万像素；整批校验后才调用识图 | 已不只是检查文件扩展名与 MIME |
| OCR | 简版文字/位置协议，兼容对象或消息数组；格式错误、空输出、截断才重新看同图一次 | 超时、认证、限流不会自动触发这次恢复；恢复成功仍需校对 |
| 超时 | 视觉独立预算；评审另有排队、阶段、总预算 | 不应把一个配置理解成所有流程的统一时限 |
| 关系规则 | SKILL 正文和 YAML 规则实际进入上下文 | 这些是提示约束，不是模型训练 |
| Embedding | 支持独立配置、原生千问适配、严格报错或显式词面降级 | 不再默认继承聊天密钥；多模态向量模型在本项目目前仍接收文本 |
| 只读评审 | 服务端创建 batch_id / candidate_id 并保存原文，模型返回 ID 与评价，服务端回填原文 | 不再要求模型逐字抄写 evaluated_text；不是持久化候选版本号 |
| 评审模式 | 低风险无天气证据可 quick；风险场景、手动详细评估及复评 full | quick 缩短规范和预算，并不必然更换模型 |
| 部分评价 | 保留有效条目，但整组 partial 不推荐、不启动修订 | 有部分有效结果，不代表整组通过 |
| 候选建议 | 完整有效评价中，比较可用候选的核心维度短板，再比较合计，允许并列 | 不制造唯一最佳，不把五档判断叫准确率 |

## 当前架构地图

~~~mermaid
flowchart TD
    A[文字背景] --> D[组装本轮上下文]
    B[上传截图] --> C[图片校验与识图 / 人工校对]
    C --> D
    E[联系人摘要 / 检索片段 / 本人风格 / 关系规则] --> D
    D --> F[联合理解与生成 / 本地结构校验]
    F <-->|按需| G[受限工具调用 / MCP 天气]
    F --> H[返回三条候选与任务状态]
    F --> I[内存质量任务 / quick 或 full]
    I --> J[必要时一次修订 / full 复评]
    F --> K[可选 SQLite 记忆任务]
    K --> E
    H --> L[前端轮询 / 编辑保护 / 显式采用改进版]
    J --> L
~~~

核心栈：原生 HTML/JavaScript、FastAPI/Uvicorn、Pydantic、SQLAlchemy/SQLite、OpenAI 兼容 SDK、httpx、MCP Python SDK、Pillow；本地 sentence-transformers 是可选检索依赖。工作流通过 Python 服务显式编排。

## 第 1 天：请求主链、数据契约与多模态输入

**当天目标：拿到一条用户输入后，能指出它经过哪些函数、哪些外部调用，以及在哪里可能失败。**

| 时间 | 操作主题 | 必须留下的结果 |
| --- | --- | --- |
| 00:00–00:50 | 环境与全局地图 | 测试基线、模块职责表 |
| 00:50–01:40 | API 与请求/响应模型 | 一份请求字段与响应字段笔记 |
| 01:40–02:30 | 上下文与结构化生成 | Prompt 来源表、校验层次 |
| 02:30–03:20 | 上传与 OCR 恢复 | 输入失败处理表 |
| 03:20–04:10 | 用测试追踪四条路径 | 四份“预测—断言—实现”记录 |
| 04:10–05:00 | 画图与口述 | 3 分钟讲解、当天自测 |

### 1.1 环境与全局地图

1. 在项目终端执行以下命令。已有环境缺依赖时先确认解释器是否正确，暂不升级依赖。

~~~bash
cd /Users/jiaqihong/Desktop/QY_Chat
conda activate my_demo1
python --version
python -B scripts/run_tests.py
node tests/test_frontend_quality_async.cjs
node tests/test_frontend_quality.cjs
node tests/test_frontend_topic.cjs
node tests/test_frontend_memory.cjs
~~~

若当前终端不能激活 conda，可将下文 Python 命令的解释器换为本机已核实存在的 /opt/anaconda3/envs/my_demo1/bin/python，不需要重新安装 Python。预期后端结束为 OK；当前快照为 290 项。前端四个命令各输出通过提示。以后代码变化时，数量可以改变。

2. 看 [现有阅读路线](/Users/jiaqihong/Desktop/QY_Chat/docs/code-reading-guide.md)，最多花 10 分钟；再浏览 [main.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/main.py:280) 的路由名称。
3. 在笔记中列出五层：页面交互、HTTP 接口、业务编排、模型/工具访问、数据存储。每层写出两个实际文件名。

检查点：解释清楚“AI 回复辅助”这一产品目标：系统生成候选，由人选择；没有自动给对方发微信的执行链。

### 1.2 API 与数据契约

依次阅读 [create_reply_from_confirmed_chat](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/main.py:440)、[PreparedReplyRequest](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/schemas.py:274)、[ReplyResponse](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/schemas.py:394)、[get_db](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/database.py:25)。

1. 跟踪 user_id、contact_id、relation_type、background_context、extracted_chat、controls、save_to_memory 这七组输入。
2. 跟踪 replies、quality_job_id、quality_status、memory_job_id、memory_status、retrieval 这六组响应。
3. 找出输入校验、业务异常转 HTTP 错误、数据库会话关闭的位置。
4. 比较同步路由与截图异步路由：找出 run_in_threadpool 的调用点，说明哪些工作是同步阻塞的。

用自己的话写清：Pydantic 负责数据契约；SQLAlchemy 负责数据库映射和会话；一次 HTTP 返回成功不代表两个后台任务已经成功。

必要时只查 [FastAPI 异步与并发文档](https://fastapi.tiangolo.com/async/) 中关于路径函数和普通辅助函数的说明。框架可在线程池运行同步路径函数；在 async 函数中直接调用普通阻塞函数不会自动变成非阻塞。

### 1.3 上下文与结构化生成

阅读 [_build_reply_response](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/main.py:793)、[_load_generation_context](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/main.py:986)、[understand_and_generate](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/generation_service.py:82)、[GenerationPayload](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/generation_service.py:43)。

1. 按调用顺序画出：加载关系规则 → 标准化本轮消息/背景 → 加载记忆与风格 → 联合理解生成 → 评审入队 → 可选记忆入队 → 响应。
2. 给每类上下文标来源：本轮原文、用户概括背景、历史事实、本人风格、规则。找出历史主题没有混入本轮检索查询的位置。
3. 对照模型输出字段：topic、analysis、scenario、strategy、三条 replies。普通首次生成主要合并为一次模型调用；工具往返、评审和记忆另计。
4. 标出三个检查层：上游结构化请求选项 → JSON 解析 → 本地 Pydantic/业务约束。举一个“格式正确但事实错误”的例子。
5. 浏览 [skill_loader.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/skill_loader.py:20)，确认关系规则正文如何进入上下文。

检查点：解释“我已经有约了，无法赴约”和“如果临时需要可以再联系”为什么不能被压缩成“我答应参加”。主题、情绪推断、回复策略是不同信息。

### 1.4 图片上传与 OCR 恢复

阅读 [save_screenshot](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/upload_service.py:36)、[extract_chat_batch](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/main.py:294)、[extract_chat_from_image](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/vision_service.py:86)、[_parse_extracted_chat](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/vision_service.py:286)。

按顺序检查：分块读取 → 大小检查 → 实际解码 → 按解码类型确定后缀 → 整批验证 → 顺序识图 → 相邻截图边界合并 → 清理临时文件。

抄下并解释这张表：

| 情况 | 代码应有的行为 | 理由 |
| --- | --- | --- |
| 第二张图片损坏 | 尚未调用第一张图片的模型 | 避免对注定失败的整批请求先产生调用 |
| OCR 返回空串/坏结构/截断 | 同一图片最多重新识别一次 | 有界恢复，不能拿坏输出当聊天 |
| 识图超时/认证失败/限流 | 清晰报错，不走格式恢复 | 错误性质不同，盲目重试可能增加等待和费用 |
| 恢复成功 | 明确标记需要复核 | 合法结构无法保证无漏行、说话人正确 |

备用视觉模型未配置时，恢复轮可以再次使用首次视觉模型；“恢复”不必然意味着切换模型。简版 OCR 不强发其不支持的结构化参数。图片位置也来自模型，不能声称是可靠视觉检测器的实测坐标。

有余力再读 merge_extracted_chats：为什么只对相邻截图的前后边界去重，而不能删除全局所有重复短句。

### 1.5 四个精读测试

全套测试已经跑过，这一时段主要练习读断言；需要重跑可直接使用第一时段的隔离入口。

| 进入的测试 | 先预测什么 | 应找到的证据 |
| --- | --- | --- |
| [test_v021_pipeline.py:38](/Users/jiaqihong/Desktop/QY_Chat/tests/test_v021_pipeline.py:38)，test_one_strong_call_receives_history_style_background_and_controls | 生成模型会收到哪些上下文 | 历史、风格、背景与控制项确实传入 |
| [test_structured_generation.py:111](/Users/jiaqihong/Desktop/QY_Chat/tests/test_structured_generation.py:111)，test_broken_json_is_rejected_without_guessing_or_logging_private_text | 坏 JSON 是否被补成正常结果 | 抛错，诊断不输出私人原文 |
| [test_demo_readiness.py:83](/Users/jiaqihong/Desktop/QY_Chat/tests/test_demo_readiness.py:83)，test_later_invalid_file_is_detected_before_first_model_call | 前一张是否已调用模型 | 模型调用次数为零 |
| [test_vision_recovery.py:66](/Users/jiaqihong/Desktop/QY_Chat/tests/test_vision_recovery.py:66)，test_invalid_empty_and_truncated_primary_recover_from_same_image_once | 恢复的输入和次数 | 仍是原图，最多两轮尝试，结果待复核 |

每题记录：输入如何构造、patch 替换了什么、哪个断言证明目标、源码由哪一分支实现。

### 1.6 当天交付与验收

- [ ] 画请求时序图，至少标出截图可选、工具可选、两条后台支线。
- [ ] 不看代码，3 分钟说明 main / schemas / services / models 各自职责。
- [ ] 回答：async 为什么不保证更快？Schema 为什么不保证事实正确？为什么识图后仍需人工确认？
- [ ] 用一句话指出当前整组生成入口；不要根据 analysis_service 等历史文件名猜主链。

答案要点：阻塞 SDK 需要正确调度；结构和语义是两类约束；文字识别、角色归属和完整性需要分别验证。

## 第 2 天：RAG、个性化记忆与事务一致性

**当天目标：能从一次检索讲到向量、排序、隔离、失效策略，并解释后台记忆为什么不会部分成功或被迟到结果恢复。**

| 时间 | 操作主题 | 必须留下的结果 |
| --- | --- | --- |
| 00:00–00:50 | 表结构与上下文来源 | 数据职责/隔离维度表 |
| 00:50–01:40 | Embedding 适配 | 请求批次图与失败策略 |
| 01:40–02:30 | 检索与时间衰减 | 手算排序、召回边界 |
| 02:30–03:20 | 持久化记忆任务 | 事务和取消时序图 |
| 03:20–04:10 | 六个测试与参数实验 | 一份 RAG 实验记录 |
| 04:10–05:00 | 口述与方案取舍 | 5 分钟 RAG 深挖回答 |

### 2.1 数据职责与隔离

阅读 [models.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/models.py)、[_load_generation_context](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/main.py:986)、[style_service.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/style_service.py:62)。

1. 将 ContactMemory、MemoryEpisode、ConversationMemory、UserStyleProfile、StyleExample、ReplyFeedback、MemoryJob 各写一句职责。
2. 标出用户/联系人事实隔离与用户/关系风格共享的区别；找出联系人唯一约束与历史片段联合索引。
3. 追踪自动风格学习：只取本人逐条消息；概括背景和对方消息为什么不应进入本人风格。
4. 查清部分 ID 关联是普通字段、不是数据库外键；删除逻辑依赖服务层。

检查点：能解释为什么“联系人长期事实”“可召回的事件片段”“本人语气”需要分开；这是应用存储与提示增强，不是训练模型记忆。

### 2.2 Embedding 配置与原生接口适配

阅读 [embedding_options](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/embedding_service.py:66)、[semantic_similarities](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/embedding_service.py:109)、[_dashscope_multimodal_vectors](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/embedding_service.py:210) 和 [配置指南](/Users/jiaqihong/Desktop/QY_Chat/docs/semantic-memory-setup.md)。只需看配置名称与代码，不需要打开真实密钥。

1. 画出 local / openai_compatible / dashscope 三个分支。
2. 解释独立凭据及复用条件：只有显式允许、且地址匹配时才能回退复用聊天凭据。
3. 找到 dashscope + qwen3-vl-embedding 的特殊适配；本项目提交的是文本。解释为什么 enable_fusion=false：需要逐段独立向量。
4. 计算一次查询 + 60 条历史共有 61 段文本，按 20 段分批应为 20、20、20、1。找到服务端乱序结果怎样按 index 还原。
5. 找出向量条数、维度、有限数值、零范数等校验，解释错误向量为什么不能静默参与排序。

检查点：区别“调用配置存在”“API 连通”“返回合法向量”“召回符合业务”四个层次。

### 2.3 检索排序与手算实验

阅读 [retrieve_relevant_memories](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/retrieval_service.py:65) 和 [_time_decay_factor](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/retrieval_service.py:292)。

按顺序抄出：按 user_id/contact_id 筛选 → 最近最多 60 条 → 排除当前重复聊天 → 查询/历史编码 → 相似度 → 时间衰减 → 门槛过滤 → 排序 → 摘要去重 → 默认最多 3 条 → 拼接上下文。

公式：score = similarity × 2 ** (-age_days / half_life_days)。先手算，再运行下列纯本地实验：

~~~bash
python - <<'PY'
memories = [("A", 0.80, 30), ("B", 0.60, 0), ("C", 0.90, 90)]
for half_life in (30, 90):
    ranked = [(name, similarity * 2 ** (-age / half_life))
              for name, similarity, age in memories]
    print(half_life, sorted(ranked, key=lambda item: item[1], reverse=True))
PY
~~~

预期：半衰期 30 天，B=0.60、A=0.40、C=0.1125；半衰期 90 天，A≈0.635、B=0.60、C=0.45。当前语义模式门槛为 0.16，所以前一组的 C 会在排序返回前被过滤。这个小程序只演示排序，不执行项目的筛选和去重。

再回答四题：重要旧事实怎样避免被时间衰减误伤？最近 60 条之外的事件为什么连评分机会都没有？换向量模型后阈值还能照搬吗？为什么向量每次重算会影响成本与延迟？

### 2.4 记忆后台任务与原子写入

阅读 [process_next_memory_job](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/memory_queue_service.py:70)、[AtomicMemorySession](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/memory_queue_service.py:44)、[data_management_service.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/data_management_service.py)。

1. 画 pending → running → completed / failed / cancelled；标出启动恢复 running → pending。
2. 找到条件更新认领任务，以及模型请求前关闭会话的位置。
3. 找到模型返回后重新检查 running 的条件更新；在旁边写出删除/手动修改发生在网络等待期间的时序。
4. 解释 commit 被改为 flush 的局部适配：内部服务先刷新，最外层一次提交联系人、会话、事件片段、风格和任务状态；异常时整体回滚。
5. 解释为什么网络调用不应占据长数据库写事务，以及为什么后台线程使用独立 Session。

用 [SQLAlchemy Session 文档](https://docs.sqlalchemy.org/en/20/orm/session_basics.html) 核对 flush、commit、rollback 与线程会话边界。准确表述是本地事务与任务状态约束；崩溃恢复可能重复模型调用，不能声称端到端 exactly-once。

### 2.5 六个精读测试

| 测试入口与方法 | 预测和验证重点 |
| --- | --- |
| [test_semantic_rag.py:40](/Users/jiaqihong/Desktop/QY_Chat/tests/test_semantic_rag.py:40)，test_dedicated_credentials_are_used_without_inheriting_chat_secret | 不隐式继承聊天凭据 |
| [test_semantic_rag.py:125](/Users/jiaqihong/Desktop/QY_Chat/tests/test_semantic_rag.py:125)，test_strict_mode_does_not_replace_vector_failure_with_lexical_scores | error 模式明确失败；与 lexical 降级作对照 |
| [test_semantic_rag.py:158](/Users/jiaqihong/Desktop/QY_Chat/tests/test_semantic_rag.py:158)，test_retrieved_memory_reaches_the_strong_generation_prompt | 召回不是只显示在页面，确实进入生成上下文 |
| [test_qwen_embedding.py:77](/Users/jiaqihong/Desktop/QY_Chat/tests/test_qwen_embedding.py:77)，test_native_batches_twenty_texts_and_preserves_cross_batch_order | 分批与顺序正确 |
| [test_retrieval_service.py:187](/Users/jiaqihong/Desktop/QY_Chat/tests/test_retrieval_service.py:187)，test_semantic_similarity_is_multiplied_by_age_without_bonuses | 相似度乘衰减，没有额外关系奖励 |
| [test_v021_pipeline.py:212](/Users/jiaqihong/Desktop/QY_Chat/tests/test_v021_pipeline.py:212)，test_delete_while_model_running_does_not_resurrect_contact | 迟到结果不恢复已删除数据 |

最后写一张诊断表：无历史为 not_scored；成功计算但零召回仍可为 semantic；向量失败后回退为 lexical_fallback；严格模式失败报错。不能仅凭返回空列表判断有没有使用向量。

### 2.6 当天交付与验收

- [ ] 5 分钟讲清一次检索，包括候选池、编码对象、相似度、衰减、门槛、去重、注入。
- [ ] 手算两种半衰期的排名并解释变化。
- [ ] 画“用户删除记忆，同时后台模型返回”的时序图。
- [ ] 回答：为什么这里有 RAG 却没有向量数据库？风格为什么不是微调？严格失败和降级分别适合什么场景？
- [ ] 提出一个改进：持久化向量并记录模型版本，按增量更新；说明这属于待实现设计。

## 第 3 天：模型适配、MCP 与最新质量评审协议

**当天目标：能讲清模型如何使用工具，以及怎样把不稳定的模型评价变成可校验、可绑定、可部分保留的业务结果。**

| 时间 | 操作主题 | 必须留下的结果 |
| --- | --- | --- |
| 00:00–00:50 | 模型角色、预算和异常 | 调用边界表 |
| 00:50–01:40 | MCP 工具往返 | 工具调用时序图 |
| 01:40–02:30 | 候选 ID 绑定协议 | 五种错配结果预测 |
| 02:30–03:20 | quick/full 与可用性判断 | 分流表与推荐手算 |
| 03:20–04:10 | 协议和分级测试 | 六份断言记录 |
| 04:10–05:00 | 工程故事与口述 | 评审改造故事、当天自测 |

### 3.1 模型角色与配置边界

阅读 [llm_client.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/llm_client.py:23)，重点为 model_role、configured_models、model_request_timeout、quality_model_limits、_structured_output_options、_client。

1. 列出视觉、生成、评审、记忆角色的调用方；说明“角色独立”与“型号独立”的区别。
2. 找出 ContextVar 的 set / reset，解释为什么上下文退出必须恢复，即使发生异常也要恢复。
3. 区分通用请求预算、视觉独立预算、质量 quick/full 用量预算。预算不是固定响应时长承诺。
4. 找到 provider 专有参数的发送条件，解释不能把一家供应商的参数无条件传给所有兼容接口。
5. 找到 SDK 自动重试设置，以及项目自己的有界恢复/修订逻辑；避免把多层重试叠加理解成“最多一次调用”。

自己填一张表：调用角色 → 模型来源 → 允许的工具 → 预算来源 → 错误如何上报。注意 ContextVar 能管理上下文值，但不能推导出任意新建线程都自动继承；检查实际进入线程的代码在哪建立 trace 与角色。

### 3.2 MCP 与模型工具调用

阅读 [_complete_text](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/llm_client.py:118)、[mcp_service.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/mcp_service.py:33)、[mcp_server.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/mcp_server.py:16)、[weather_contract.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/weather_contract.py)。

1. 按实际消息顺序画图：应用给模型工具定义 → 模型返回 tool_calls → 应用检查工具名/参数 → MCP stdio 会话 initialize/list_tools/call_tool → 天气结果 → role=tool 返回模型 → 最终结构化回复。
2. 解释两个边界：Function Calling 让模型表达调用意图；MCP 规范应用与工具服务之间的通信。模型不直接运行本机函数。
3. 找到最多三轮模型请求、最多两次天气工具执行机会、最后一轮禁用继续调用的代码。
4. 检查天气证据如何进入评审，但不作为原始工具结果直接加入长期记忆载荷。
5. 浏览缓存和城市日期逻辑：TTL、跨当地日期失效、城市歧义、部分缺失如何表达。

精读 [test_weather.py:191](/Users/jiaqihong/Desktop/QY_Chat/tests/test_weather.py:191) 的 test_tool_result_is_sent_back_to_same_model_before_final_response。写出第一次与第二次模型请求的 messages 有何差别。再看同文件的工具预算测试，解释未知工具和坏参数为什么不能执行。

检查点：工具失败应该作为失败证据回传；没有温度或空气质量数据不能填零。普通聊天不会为了“判断需不需要天气”再新增一轮独立模型调用。

### 3.3 候选 ID 绑定协议

阅读 [ReadOnlyReviewContract](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/quality_contract.py:15) 和 [review_candidates](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/quality_service.py:126)。

按这六步追代码：

1. 服务端将原文保存为 tuple。
2. 本次创建 batch_id，并构造 batch_id:index 形式的 candidate_id。
3. Prompt 发送候选 ID 和文本；评审返回 ID、五维档位、依据和问题列表。
4. bind 按 ID 匹配，不按数组位置猜测；组评价另核对 batch_id。
5. 服务端回填 evaluated_text，再做评价内容校验。
6. 保留通过校验的条目；存在协议错误且仍有有效条目时，整组标记 partial；没有任何有效条目时标记 invalid_response，后台任务失败。两种情况均不推荐、不进入修订。

闭卷预测五种情况：三条评价换序；重复一个 ID；使用上次评审的 ID；一条缺依据；组 batch_id 错误。核对代码：换序可以按 ID 还原，重复 ID 的相关条目全部无效，旧 ID 不接受，缺项可局部失败，组失配不能取得整组通过。

用下面的纯本地实验观察“换序”和“重复”的不同；不读取配置、不调用模型、不写数据库：

~~~bash
python -B - <<'PY'
from chat_reply_agent.services.quality_contract import ReadOnlyReviewContract

contract = ReadOnlyReviewContract(["好，下次再聊", "明白，你先忙", "收到，早点休息"], {0: [], 1: [], 2: []})
data = contract.example()
data["items"].reverse()
rows, group, errors = contract.bind(data)
print("重排后绑定索引：", [row["candidate_index"] for _, row in rows])
print("绑定错误：", errors)
data["items"][1]["candidate_id"] = data["items"][0]["candidate_id"]
rows, group, errors = contract.bind(data)
print("重复后保留条目数：", len(rows))
print("错误类型：", sorted({item["type"] for item in errors}))
PY
~~~

预期第一次索引为 [2, 1, 0]、绑定错误为空；第二次只保留一条，错误类型包含 duplicate_candidate_id 和 missing_target。此实验只验证绑定，不代表示例中的 null 分数能通过后续质量判断。

### 3.4 分级评审与候选推荐

阅读 [select_review_mode](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/quality_queue_service.py:340)、[load_quality_skill](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/skill_loader.py:40)、[_build_quality_item](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/quality_service.py:282)、[_build_report](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/quality_service.py:363)。

1. 写出分流：低风险且无天气证据可 quick；风险、道歉、边界、明确拒绝/降温、说话人待确认、天气依据等使用 full；手动详细评估和修订后的复评始终 full。
2. 比较 quick/full 的规范、每维依据长度和模型预算。不要把精简评审解释成跳过评审。
3. 区分 issues / hard_issues / critical_unknowns / advisories；普通提醒和缺个人风格样本不应单独否定其他有效维度。
4. 区分自然度、相关性、关系适配、场景适配、风格适配五维。四个核心维度缺失、关键未知和明确硬伤的语义不同。
5. 在完整有效评价中，只从 usable 条目里推荐；先比较核心维度最小值，再比较总和。并列可以推荐多条；推荐一条也不等于整组全通过。

手算：A=[5,5,5,4]，B=[4,4,4,4]，C=[5,5,5,3]。假设没有其他问题、评价完整：A 和 B 都可用，A 因合计更高优先；C 有低于 4 的已知维度，应提示调整，不能因总和高就胜出。再把 B 改成 [5,5,4,5]，解释为何可以并列。

### 3.5 六个协议与分级测试

| 测试入口与方法 | 预期 |
| --- | --- |
| [test_quality_binding.py:25](/Users/jiaqihong/Desktop/QY_Chat/tests/test_quality_binding.py:25)，test_quotes_newlines_and_reordering_bind_without_echoing_original | 引号、换行、重排不依赖模型抄原文 |
| [test_quality_binding.py:93](/Users/jiaqihong/Desktop/QY_Chat/tests/test_quality_binding.py:93)，test_partial_review_is_retained_and_never_starts_repair | 保留有效评价，修订调用为零 |
| [test_quality_binding.py:119](/Users/jiaqihong/Desktop/QY_Chat/tests/test_quality_binding.py:119)，test_partial_recheck_keeps_original_review_and_does_not_publish_revision | 复评不完整，保留原稿评价，不发布改进版 |
| [test_quality_usefulness.py:79](/Users/jiaqihong/Desktop/QY_Chat/tests/test_quality_usefulness.py:79)，test_recommendation_prioritizes_weakest_core_then_sum | 验证短板优先排序 |
| [test_quality_usefulness.py:88](/Users/jiaqihong/Desktop/QY_Chat/tests/test_quality_usefulness.py:88)，test_routing_conservatively_keeps_risks_in_full_mode | 风险与天气进入 full |
| [test_quality_usefulness.py:122](/Users/jiaqihong/Desktop/QY_Chat/tests/test_quality_usefulness.py:122)，test_quick_original_uses_full_recheck_and_preserves_useful_summary | 初评 quick、复评 full |

只读评审、定向修订、复评是不同调用。初评 partial 直接结束；完整评价中的明确缺陷/重复作用等才可能触发一次修订。未知分不是低分，失败不会无限循环。

### 3.6 当天交付与验收

- [ ] 画清模型消息协议与 MCP 协议两段往返。
- [ ] 解释为什么用随机批次 ID，而不是依赖模型抄原文或按返回位置匹配。
- [ ] 用“问题 → 方案 → 代价 → 测试证据 → 局限”讲 3 分钟评审协议改造。
- [ ] 回答：同一型号做独立评审还会有偏差吗？partial 为什么不自动修订？quick 为什么不能声称固定提速比例？

答案要点：ID 解决本次结果关联，不证明判断正确；部分失败意味着信息不足以驱动整组后续动作；同型号仍可能有相关偏差；单次时延差不能当统计性性能保证。

## 第 4 天：并发可靠性、评测与面试演练

**当天目标：把前三天的实现连成可讲述的工程方案，并用故障、指标和扩展设计接住追问。**

| 时间 | 操作主题 | 必须留下的结果 |
| --- | --- | --- |
| 00:00–00:50 | 质量队列与超时取消 | 队列状态图、预算算例 |
| 00:50–01:40 | 前端竞态与回归测试 | 编辑/迟到结果时序 |
| 01:40–02:30 | 可观测性与效果评测 | 指标表与对照评测方案 |
| 02:30–03:20 | 扩展与基础知识迁移 | 一页系统设计 |
| 03:20–04:10 | 演示与模拟追问 | 一次录音、三处漏洞 |
| 04:10–05:00 | 针对性补缺与最终验收 | 项目讲稿和自测评分 |

### 4.1 队列、超时和取消

阅读 [QualityJobManager](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/quality_queue_service.py:96)，重点 submit、_prune、_begin_stage、_review、_run、_fail、cancel。

1. 画 pending/queued → running/review → 可选 running/repair → running/recheck → completed；另画 failed、timed_out、cancelled。
2. 标注默认容量：2 个工作线程、16 个活动或未收尾任务、64 个保留结果、终态 TTL 600 秒。内存任务重启即失。
3. 找到 RLock、deepcopy 和请求指纹，分别解释状态保护、输入快照、重复提交复用。
4. 找出“任务已终态但 future 仍未完成”仍占容量的判断，说明这样如何避免反复点击堆积网络请求。
5. 比较记忆任务的 SQLite 持久化与评审任务的内存生命周期，说明为什么不能直接多 worker 扩容。

手算预算：排队 20 秒，实际开始后总预算 240 秒，每阶段最多 90 秒。初评用 70 秒、修订用 85 秒，则开始复评时总预算剩 85 秒，复评最多 85 秒。排队的 20 秒不扣减这个 240 秒，但属于用户等待时间；实际请求仍可能更早失败。

精读 [test_quality_queue.py:121](/Users/jiaqihong/Desktop/QY_Chat/tests/test_quality_queue.py:121) 的 test_cancel_queued_and_running_jobs_rejects_late_results，以及同文件的 test_separate_timeouts_keep_unfinished_work_bounded。

Future.cancel 无法取消已经开始运行的调用；代码通过状态检查拒绝迟到结果，而非保证上游请求和费用撤销。参阅 [Python Future 文档](https://docs.python.org/3.12/library/concurrent.futures.html#concurrent.futures.Future.cancel)。此外，原稿已评完而修订失败时，任务可能仍 completed，同时 repair_status=failed；必须同时理解任务、阶段与质量三个状态。

### 4.2 前端竞态与原稿保护

阅读 [applyQualityResult](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/static/index.html:2092)、[pollQuality](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/static/index.html:2129)、[useQualityRevision](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/static/index.html:1970)。

按如下事件画时序，并在代码中找到每个检查点：

1. 提交候选组 A，开始轮询任务 A。
2. 用户改了第一条文字，或提交了新候选组 B。
3. 任务 A 结果迟到。
4. 前端检查当前响应对象、轮询 token、job_id、提交文本、当前输入值和编辑状态。
5. 评价只更新仍匹配的区域，不覆盖用户原稿和光标。
6. 只有满足复评条件且用户显式点击，才采用改进版；采用单条后旧组内差异判断失效。

运行四组前端命令，再读 [test_frontend_quality_async.cjs](/Users/jiaqihong/Desktop/QY_Chat/tests/test_frontend_quality_async.cjs)。先找测试构造的“编辑后迟到结果”，再对照真实函数。

若本机浏览器测试环境已配置，可用本时段剩余时间运行 node tests/test_frontend_quality_browser.cjs；没有依赖就读测试场景，不为此占用后续时间。该浏览器测试模拟 HTTP，不证明真实模型输出质量。

检查点：高风险页面等待仅是展示策略，API 已返回草稿，不能介绍成服务端安全拦截。

### 4.3 可观测性与效果评测

阅读 [telemetry_service.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/telemetry_service.py:17)、[evaluation_service.py](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/services/evaluation_service.py:86)、[evals/dataset.yaml](/Users/jiaqihong/Desktop/QY_Chat/chat_reply_agent/evals/dataset.yaml)。

1. 找 trace_scope、timed_stage、current_trace、请求 ID、后台父请求 ID，解释一次 HTTP 请求和后续任务怎样关联。
2. 找日志记录的耗时、错误类型、校验字段路径；说明为什么不需要把聊天正文和密钥打印出来才能定位格式失败。
3. 区分已有指标和未来指标：代码已有阶段耗时与质量调用用量；完整成本核算、压力测试和跨请求 P95 需要另行采样与统计。
4. 画一份 12 个合成用例的评测矩阵：普通收尾、拒绝/边界、信息不足、截图角色问题、工具失败、记忆干扰各 2 例。这里只设计用例和标签，不要求在本时段执行付费评测。
5. 留出一部分未参与提示词调试的样例，设计盲评比较：事实/承诺边界、相关性、自然度、候选差异、人工偏好。

| 指标 | 如何解释 | 不能推出什么 |
| --- | --- | --- |
| 首次回复耗时 | 请求到草稿返回；必要时另记页面实际可见时间 | 不等于评审与记忆全部完成 |
| 评审排队/执行耗时 | 分开统计，定位资源排队或上游慢 | 异步并不使模型推理本身变快 |
| 格式有效率 | 满足协议和本地校验的比例 | 不等于事实正确率 |
| 检索 Recall@k | 人工标注的相关片段被召回的比例 | 相似度高不自动等于有用 |
| 人工偏好 | 固定规则、盲评、保留集上的选择 | 评审模型高分不保证对方满意 |
| P50/P95、费用 | 需要多次可比样本及完整调用口径 | README 的单次样本不等于稳定指标 |

### 4.4 一页扩展设计与基础知识迁移

用这张表写一页方案，每项只写触发条件、改动、代价和验证指标，不实现大改造：

| 触发条件 | 建议演进方向（当前未实现） | 必须回答的取舍 |
| --- | --- | --- |
| 多实例服务 | 共享任务存储、可恢复工作队列、租约和幂等键 | 重复投递、worker 崩溃、迟到结果怎样处理 |
| 大量历史记忆 | 持久化向量、增量编码、模型版本、合适的索引 | 重建成本、召回质量、隔离与删除一致性 |
| 上游限流或费用压力 | 并发/配额限制、调用预算、缓存、按需降级 | 新鲜度、质量、成本、尾延迟怎样权衡 |
| 对外提供服务 | 服务端认证与授权、代理请求体限制、数据库迁移、数据生命周期管理 | user_id 不能充当登录凭证，上传业务校验不等于入口限流 |

随后把项目问题映射到基础知识：HTTP 状态与连接超时；协程/线程/GIL 与 I/O 等待；事务、索引、唯一约束；缓存失效、幂等与重试；Embedding、上下文预算、RAG 与微调的区别。

优先能结合源码举例。Transformer、LoRA、分布式训练可先准备概念与适用场景；若岗位明确要求模型训练，这 20 小时应用工程计划需要另补训练内容。

### 4.5 演示与模拟面试

先用 10 分钟排练以下合成故事，避免临场编输入：

> 朋友问今晚要不要一起打游戏。我回复已经约了别人、今晚来不了。对方说好，那下次吧。我想简短自然地收尾，不承诺具体哪一天。

演示顺序：输入和产品边界 → 当前主链图 → 三候选/状态的结构 → 编辑后旧评价失效 → 联系人记忆与检索 → 一个失败处理测试。没有本次真实运行的结果时，使用测试解释行为，明确是模拟；不要把预设文本当作实时模型输出。

需要真实页面演示时，优先复用已有可用环境并使用合成身份和临时数据。真实模型调用会产生费用，不属于本计划必做项。独立演示数据库与截图素材的现成步骤在 [展示检查报告](/Users/jiaqihong/Desktop/QY_Chat/docs/interview-readiness.md)；该报告含历史行为，质量机制以本手册核对的当前代码为准。

接着录音完成一轮 30 分钟模拟：

- 3 分钟：项目介绍和你的实际贡献。只陈述自己能证明、能解释的工作。
- 7 分钟：追踪一条请求，再讲 RAG。
- 8 分钟：深挖评审协议、partial、修订和取消。
- 7 分钟：假设模型超时、用户删除、服务重启，逐项处理。
- 5 分钟：容量提升与成本/质量权衡。

余下 10 分钟回听，记下三处最含糊的回答，进入最后时段修补。

### 4.6 最终补缺与验收

用 20 分钟只回查刚才的三处漏洞；10 分钟写 90 秒和 3 分钟讲稿；剩余时间从下方 20 题中抽 5 题，每题答 2 分钟。

90 秒结构：15 秒用户问题 → 25 秒核心链路 → 30 秒两个技术难点 → 20 秒证据与当前局限。3 分钟版增加一个失败案例和一个可验证的取舍。不要背所有类名，关键是说得清数据流与判断依据。

验收：抽问的每题都能说出机制、一个源码入口、一个测试证据和一个限制。若做不到，优先补这些题，暂不增加新功能。

## 20 道高频追问与答案检查点

先盖住右列回答，再核对。每题按 0–2 分自评：0 分讲不清；1 分能讲机制；2 分能给代码、测试与取舍。32/40 可作为自己的复习目标，不是面试通过概率。

| 问题 | 回答至少覆盖 |
| --- | --- |
| 1. 一次请求怎么走？ | 输入/校对、上下文、联合生成、工具可选、两种后台任务、前端轮询 |
| 2. 为什么不用 LangChain/LangGraph？ | 当前固定流程显式编排便于约束调用与事务；更复杂恢复/分支可评估框架，比较迁移代价 |
| 3. async 为什么不一定快？ | 阻塞 SDK、线程调度、I/O 与 CPU 区别；并发不等于单次推理提速 |
| 4. 为什么生成结果还要校验？ | 供应商能力差异、截断、字段/枚举/空文本；结构正确不保证事实 |
| 5. OCR 恢复为什么不能一律重试？ | 结构类与网络/权限类错误不同；最多一次同图恢复；校对与完整性 |
| 6. RAG 与微调有什么区别？ | 运行时检索/上下文增强与权重更新；本项目风格也未做训练 |
| 7. 向量库在哪里？ | 当前 SQLite 存文本；现场计算候选向量；没有持久化 ANN 索引 |
| 8. 召回如何排序和隔离？ | user/contact、60 条池、相似度乘衰减、阈值、去重、top 3 |
| 9. 向量失败怎么办？ | error 与 lexical 策略；诊断标真实方法，零结果也需要方法信息 |
| 10. 为什么分开联系人记忆和风格？ | 事实所有者与个人表达习惯不同；只学本人逐条消息，避免污染 |
| 11. 删除时后台还在跑怎么办？ | 取消旧任务、网络返回后复查、条件更新与事务，不恢复删除数据 |
| 12. 是否 exactly-once？ | 数据写入有事务约束；重启可重复调用模型；没有端到端恰好一次承诺 |
| 13. Function Calling 与 MCP 有何不同？ | 模型表达调用意图 vs 应用与工具服务通信；参数检查与有限轮数 |
| 14. 评审为什么改用 ID？ | 不依赖模型复述原文；批次隔离、重排、重复拒绝、服务端原文绑定 |
| 15. partial 为什么不修订？ | 保留已知有效评价，整组信息不完整时不驱动修订；复评要求完整 |
| 16. quick 怎么保证质量？ | 保守分流、仍保留契约校验、风险/full复评；效果还需保留集与人工评测 |
| 17. 为什么评分高不一定推荐？ | 明确硬伤/核心缺失/关键未知优先；可用候选先短板后合计；允许并列 |
| 18. 为什么取消后还占容量？ | Future 运行中不能强停；拒绝迟到结果但请求可能收尾，继续计数避免堆积 |
| 19. 如何证明变快且没变差？ | 固定输入与模型、分开阶段/队列、多个样本P50/P95、费用、盲评保留集 |
| 20. 上生产先改什么？ | 认证授权、任务持久化/多实例协调、资源限制、迁移与数据治理；按瓶颈演进 |

## 三个可准备的工程故事

| 故事 | 问题 | 当前方案 | 证据 | 应主动说明的局限 |
| --- | --- | --- | --- | --- |
| OCR 输出不稳定 | 结构合法性、漏行、说话人归属是不同问题 | 简版协议、有限恢复、独立超时、人工校对 | vision_recovery 与 demo_readiness 的错误分类/清理断言 | 格式恢复不能保证语义正确 |
| 记忆后台一致性 | 慢调用阻塞首屏，删除可能与写回竞争 | SQLite 任务、事务、返回后取消复查 | v021_pipeline 的回滚、删除、重启用例 | 本地单 worker；崩溃可重复上游调用 |
| 评审可用性与成本 | 抄原文易错、单条坏结果影响整组、评审慢 | ID 绑定、局部保留、quick/full、一次修订复评 | binding/usefulness/queue 测试 | 模型评价有偏差，真实质量与时延需统计 |

讲述时将“代码已实现”“你本次亲自验证”“准备中的改进”分开。历史验收中的单次真实耗时不能变成你的稳定性能提升结论。

## 时间不足时如何执行

仍保留每天最后的口述验收。优先完成主链、RAG、候选绑定、任务取消与事务这五项。

可压缩：CSS/页面布局、完整天气字段、全部关系规则、旧版同步分析路径、所有回归样例逐条阅读。对于今天未能跑通的环境，先以测试输入/断言完成理解，记录待办，不把它算作已实测。

可延后：引入新框架、换数据库、添加 Redis/Celery、持久化向量、模型训练、大规模部署。这些先写成可解释的设计题，四天内优先形成稳定讲解和证据。

如果某天自测未通过：次日第一个时段先用 15 分钟修补，压缩该时段的选读内容，不取消后续核心主题，不额外占用每天 5 小时。


