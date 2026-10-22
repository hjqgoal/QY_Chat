# chat_reply_agent

面向个人聊天场景的 AI 回复辅助 Agent。当前 v0.22 是本地 FastAPI Web 应用：用户可以上传 1-6 张连续微信聊天截图、只填写概括性聊天背景，或同时使用两者。截图会先经过识别和人工校对，再与补充背景一起进入语气、情绪、态度和场景分析，最终生成回复时机、回复策略和 3 条候选回复。系统只从校对后标记为“我”的逐条消息学习本人语气；对方消息和概括性背景不会进入用户风格画像。

## 面试展示基线（2026-10-23）

本轮验收：247 项后端测试、4 组前端检查及桌面/手机浏览器回归通过；真实语义 RAG、人工校对后生成、后台记忆、天气工具及评估修订流程已检查。修复了“北京市”全称导致天气漏查的问题。按用户选择保留当前 OCR，截图必须人工校对；不是免校对自动识别或生产级系统。

详见 [最终验收记录与风险](docs/release-acceptance-2026-10-23.md) 和 [代码阅读路线](docs/code-reading-guide.md)。建议据此冻结功能，先准备演示和代码讲解。

## 语义向量 RAG 配置（2026-10-23）

现在可以在真实 `.env` 独立配置 Embedding API，不会自动复用聊天密钥。百炼 `dashscope` 模式已适配你选择的 `qwen3-vl-embedding`：自动走原生多模态向量接口，为每段文字分别生成向量，现有兼容基础地址无需修改。保留 `text-embedding-v4` 等兼容接口模型和本地模型选项。严格模式 `MEMORY_EMBEDDING_FALLBACK=error` 在向量失败时明确报错；允许降级时也会返回本次真实检索方式，包括零召回结果。示例文件不保存真实密钥。

配置与验收见 [语义记忆配置指南](docs/semantic-memory-setup.md)。填好后运行 `python scripts/check_embedding_connection.py --live`，仅用合成文本验证向量、排序和上下文注入；有费用，不读写个人数据库。未加 `--live` 仅检查配置。只有真实检测通过后才能说 API 已连通。无需数据库迁移、新增依赖或本地模型下载；当前仍是按请求计算向量，不是持久化向量索引。

## 面试前检查（2026-10-22）

最新验收与演示步骤见 [面试展示检查报告](docs/interview-readiness.md)。本轮补充上传大小/解码校验，确保关系 Skill 正文实际加载；OCR 异常不再当聊天文本继续处理，识别自报分数与语义检索降级状态也作了明确提示。

新增依赖 `Pillow==13.0.0`；本次 `my_demo1` 已安装，其他环境需重新安装 `requirements.txt`。无需数据库迁移，不改真实 `.env` 或个人聊天数据。218 项离线测试通过；人工校对后的真实生成、后台评估和记忆保存通过，但当前视觉模型出现说话人错误及非法 JSON，尚未通过本次端到端验收。不能保证现场识图稳定，建议优先展示文字入口并保留人工校对。

以下各节记录历史版本行为，当前功能及验证边界以本节和检查报告为准。

## 评估超时与一次性修订（2026-10-20）

本节替代下方历史版本的评估流程。无新增依赖、无数据库迁移；不修改真实 `.env`，不重写个人记忆。

### 当前流程

1. 生成草稿后，**每轮都提交整组独立评估**，不是只有冲突场景才评估。普通聊天先显示未评估草稿；冲突、道歉、边界、中高风险、明确拒绝/降温或说话人未确认时，页面先隐藏候选，等任务结束再展示。服务端仍异步返回草稿，这个展示控制不是 API 安全边界。
2. 评审只看原始聊天、明确背景、关系、用户意图及相关记忆/天气依据，不看生成者的解释。每维必须有依据并逐字绑定原文。评估、修订、复评沿用 `MODEL_STRUCTURED_OUTPUT` 请求结构化输出，再由本地校验；无效结果不会补默认分。
3. 有明确缺陷、候选作用重复，或内部参考分低于 `QUALITY_REPAIR_THRESHOLD`（默认 80）时，**最多一次定向修订，再独立复评一次**。分数由有效五档维度换算；缺少本人样本时不计风格维度，核心维度缺失或关键条件不确定时不计算分数。未知/超时不等于低分，硬伤不能被平均分抵消。阈值未经人类偏好校准，不代表成功率，在线仍不展示百分制徽章。
4. 改进版单独展示，只有整组复评通过后才允许点击“使用这条改进版”。轮询不改原稿、光标或用户编辑；原稿后来有改动时不能直接套用旧改进版。显式使用一条后，原候选组的差异评价失效，可再评估当前组合。
5. 修订或复评失败时保留已完成的原稿评价，不发布未验证改进版。仍不合格时明确标记，不继续循环。**超时任务结束，不会在后台自动重试**；已发出的网络请求可能仍在收尾，但迟到结果丢弃。相同未结束任务复用，避免反复点击叠加调用。

质量流程一般一次评审，有问题时最多三次模型调用（评审、修订、复评），另计生成、记忆或天气调用。重启会丢失内存中的评估任务。只读评审角色不改字；在线队列现在可另行发起修订，离线评测仍只读。**没有更换首次生成的模型，也没有关闭其推理能力。**

### 时间与诊断

排队、单阶段、总流程分别计时。失败后页面保留已取得的耗时、排队时间、任务编号、模型用量和具体错误类型，区分请求超时、排队超时、总预算耗尽、限流、认证、输出截断或格式不合格。格式错误只记录固定字段路径与错误类型，不记录聊天、模型原文或密钥。

以下默认值已生效；需要调整时才添加到真实 `.env` 并重启：

```dotenv
QUALITY_TIMEOUT_SECONDS=90
QUALITY_QUEUE_TIMEOUT_SECONDS=30
QUALITY_TOTAL_TIMEOUT_SECONDS=240
QUALITY_REPAIR_ENABLED=true
QUALITY_REPAIR_THRESHOLD=80
QUALITY_THINKING_BUDGET=2048
QUALITY_MAX_COMPLETION_TOKENS=6144
```

- `QUALITY_TIMEOUT_SECONDS`：每个模型阶段的预算，5-300 秒，独立于通用 `MODEL_TIMEOUT_SECONDS`，不扣除排队时间。
- `QUALITY_QUEUE_TIMEOUT_SECONDS`：等待线程的预算，1-120 秒；到期未执行的任务不再调用模型。
- `QUALITY_TOTAL_TIMEOUT_SECONDS`：实际开始评估起算的总预算，5-600 秒；后续阶段使用剩余时间。
- 最后两项只作用于评估/修订/复评，且仅对代码支持的百炼 Qwen3.8-Max/Flash、Qwen3.7-Max/Plus/Flash 系列发送。其他服务不发送百炼专有参数。保留推理，用显式预算限制消耗；输出预算包含推理和正文，实际推理上限不超过总量一半以留出正文空间。
- 减小预算可能影响评审判断，需要真实样例校准，不能保证固定秒数。已经发出的上游请求不能强制撤回，取消不能保证撤销费用。

参数语义参考[百炼深度思考文档](https://help.aliyun.com/zh/model-studio/deep-thinking/)及[兼容接口文档](https://www.alibabacloud.com/help/zh/model-studio/qwen-api-via-openai-chat-completions)。原有超时的唯一原因不能从截图确定；应结合本轮诊断区查看具体阶段和错误。

### 验证与启动

本次通过 205 项后端离线测试、4 组前端检查，以及 1280px/390px 浏览器回归。真实接口先出现“约 48 秒返回但结构不合格”；补上结构化约束后，另一次合成短对话由当前 `qwen3.8-max` 在约 34.95 秒完成，三条候选均取得合法原文评价，校验错误为空。这是单次接口验证，不是对所有场景耗时或评审正确率的保证；真实模型的修订与复评整条链路尚未进行付费验收，该链路已由离线测试覆盖。

```bash
cd /Users/jiaqihong/Desktop/QY_Chat
conda activate my_demo1
python scripts/run_tests.py
node tests/test_frontend_quality_async.cjs
node tests/test_frontend_quality.cjs
node tests/test_frontend_topic.cjs
node tests/test_frontend_memory.cjs
# 停止原服务后重新启动，并刷新浏览器
python -m uvicorn chat_reply_agent.main:app --host 127.0.0.1 --port 8000
```

可选真实接口诊断：`python scripts/check_quality_model.py --live`。只发送合成短对话、调用一次当前评审模型，**会产生费用**；不读写聊天数据库、不触发修订，不打印密钥或聊天正文。未加 `--live` 时不会调用模型。

可选浏览器回归：`node tests/test_frontend_quality_browser.cjs`，需要本机已有 Playwright/Chrome，可用 `PLAYWRIGHT_MODULE` 指定库路径。模拟接口覆盖桌面/手机布局、原稿编辑保护、高风险展示等待、修订阶段、明确采用、取消和超时诊断。

## 后台只读评估（2026-10-19，历史行为）

以下为前一版记录；评估预算、修订及候选展示方式以 2026-10-20 一节为准。

- **先返回回复，再等待评估结果**：每次模型生成整组回复、单条重生成后，都提交一次整组只读评估，不再依赖低风险/高风险筛选。前台不等待评审完成，不再通过在线质检自动替换候选文字。备用占位示例不送评。
- 初始显示“后台评估中”，随后更新为“基本可用 / 建议调整 / 信息不足”等辅助判断，并给出逐维依据和候选差异。任务“完成”不等于回复“通过”；失败、取消、超时、队列满均明确显示，可点“评估当前回复”重试。评估仍可能出错，不是客观准确率，也不显示百分制总分。
- 后台结果只更新评价区域，不重绘输入框、不改候选、光标或用户编辑。任务绑定候选原文和当前请求；修改过的候选显示“已编辑，待确认”，其他未改候选仍可获得评价。手动重评使用当前文字，不重新生成，不保存记忆。
- 页面分别展示本次生成耗时和后台评估耗时（含单独的排队时间）。首次等待减少的是串行评估，不代表生成模型变快，也不是把历史记忆当成回复缓存。每轮仍多一次评审调用，会产生相应费用；共享模型服务的并发/限流仍可能影响实际延迟。
- 沿用 `OPENAI_REVIEW_MODEL_NAME`；未单独配置时使用现有强模型。评审只看原始聊天、用户背景、关系规则、相关记忆、本人风格证据、回复要求和天气依据，不看生成者的解释。无需新增模型或修改真实 `.env`。
- 评估任务仅保存在进程内存中，不写 SQLite，因此临时模式也可以评估而不持久化聊天。最多 2 个并发评审、16 个活动任务、64 个保留结果，终态任务保留最多约 10 分钟，达到容量时可能提前清理。重启后任务失效，页面提示重新评估；这不是持久化的分布式任务队列。
- `QUALITY_TIMEOUT_SECONDS` 默认 90 秒，可在 `.env` 配置为 5-300 秒，覆盖排队和评估等待。到期结果不再采用；正在进行的上游网络请求不能强制撤回，只能依赖请求超时收尾。取消同样会丢弃迟到结果，但不能保证撤销已经发生的调用费用。
- 新增 `POST /api/quality-jobs`、`GET /api/quality-jobs/{job_id}` 和 `DELETE /api/quality-jobs/{job_id}`。读取/取消需匹配 `user_id`；它只是本地账号隔离，不是互联网身份认证。继续使用本地单进程启动方式，不应直接开放到公网。
- 离线评测仍保持只读比较模式；旧的定向修订函数保留给离线回归，不在在线生成链路中执行。此前章节描述的“按需同步复核”已被本节替代。

无新增依赖，无数据库迁移，不改动个人记忆。**重启 FastAPI 并刷新页面后，重新生成一轮即可使用。** 验证使用模拟模型和隔离数据库，没有调用真实付费接口；实际评估质量、用时和费用仍需在你的模型上观察。

```bash
python scripts/run_tests.py
node tests/test_frontend_quality_async.cjs
node tests/test_frontend_quality.cjs
node tests/test_frontend_topic.cjs
node tests/test_frontend_memory.cjs
```

可选浏览器回归：本机已有 Playwright 和 Chrome 时，运行 `node tests/test_frontend_quality_browser.cjs`。可通过 `PLAYWRIGHT_MODULE` 指定 Playwright 路径、`BROWSER_CHANNEL` 选择浏览器；测试仅使用模拟 HTTP 响应，覆盖 1280px/390px 布局、输入保护、重评和取消。

## 对话主题修正（2026-10-19）

- 识图仅提取可见消息，不再让视觉模型顺带决定主题。校对页面显示“待结合校对内容与补充背景分析”，识别置信度也不代表对主题理解的把握。
- 现有强模型在同一次“理解与生成”调用中输出 `topic`，综合已校对消息与用户背景，保留核心事件、重要限制和当前进展。区分已发生、计划、条件方案和拒绝；可能的情绪留在分析中，不写成主题事实。
- 新主题在结果页标为“主题（模型概括）”，包括仅文字背景输入；信息不足时显示“暂无法概括”。不通过固定关键词把主题替换成模板，也不保证每次模型概括都正确。
- 聊天原文及本轮检索查询不再混入模型主题、识别备注或客户端缓存的“最后一条”。修改文字、说话人或背景会清除旧主题；上下文变更后需重新生成整组回复，不能用旧分析只重生成单条。
- 移除旧主题后，修正词面检索对未见查询片段的过高权重，让“火锅几点去呀”这类短追问不必依赖额外拼接主题才能召回相关记录。不改变时间衰减公式、相关性阈值或联系人隔离。
- 本轮后台记忆使用新主题；用户风格仍仅从本人消息学习。历史 SQLite 数据和已排队的旧任务不自动迁移或重写；已有错误记忆可在管理页人工校正或删除。
- 强模型结构化结果新增必填字符串 `topic`（最多 120 字，允许空字符串表示信息不足）；缺失或格式错误仍按生成失败处理，不保存本轮记忆。API 的 `extracted_chat` 增加 `topic_source: pending | context`，该字段只是展示来源，不作为事实可信度或用户确认标记。
- 将“算了算了你去玩”样例的补充背景纳入评测输入，补充主题事实边界要求。离线测试使用模拟模型，验证上下文传递、旧摘要隔离、调用次数和记忆保存；不是对真实模型语义正确率的证明。

无新增依赖，无需修改 `.env` 或迁移数据库。重启服务、刷新页面，再用原截图与背景重新生成。合理主题应表达“已有安排无法应邀，提出有条件临时补位后对方表示不用”，不要求逐字一致。

```bash
python scripts/run_tests.py
node tests/test_frontend_topic.cjs
node tests/test_frontend_quality.cjs
node tests/test_frontend_memory.cjs
```

## 回复评估修正（2026-10-18）

- 在线候选不再显示未经校准的百分制分数，改为“基本可用 / 建议调整 / 信息不足 / 已修改，待确认 / 仅本地检查”。这仍是辅助判断，不保证对方的反应。
- 新增 `chat_reply_agent/skills/reply-quality-review/SKILL.md` 和 `references/rubric.md`。复核服务实际加载正文、分档和边界例子，检查事实与承诺、情绪分寸、关系和本人风格。可编辑这两份文件后重启服务调整规范。
- 复核只评价输入原文，必须返回完全匹配的 `evaluated_text` 和逐维依据；缺项、错配或无依据不通过。修改后的文本不继承评分和评价；不额外调用模型给自己的修改“验收”。
- 不给评审传入生成者的分析结论、候选理由和风格标签。评审可以使用原始聊天、关系、用户意图、相关记忆及天气证据；未知情绪不能当事实。没有本人风格样本时，该维度留空。
- 用本地文字重叠提示可能重复的候选，在现有批量复核中判断接话作用是否相同。本地检测不是语义模型，可能漏掉用词完全不同的同义句；“没有命中”不会被显示成“已验证有差异”。
- 在线依然是无风险时零次额外复核，有目标时至多一次批量复核，没有新增串行评分调用。规范与证据会增加单次提示和输出长度，实际延迟需在你的模型上测量。
- 手动编辑或单条重生成会使旧的组内差异判断失效。历史记录不改写，旧分数显示为“历史评价，未校准”。SQLite 无需迁移，真实 `.env` 无需修改。
- 离线评测沿用只读模式，显示“参考分（未校准）”；四项核心维度缺失、有明确硬伤或关键不确定性时不计总分。缺少风格样本时仅计算四项核心维度，不能当成五项都已验证。评分通过还要求候选组有实质差异。
- 用户提供的“算了算了你去玩”已加入核心评测集，关系按朋友作测试假设。它用于回归，不是唯一正确答案，也不应同时用作独立的保留测试集。后续需要持续积累人工偏好和未参与提示词编写的样例，才能判断评分与人的偏好是否一致。

本次无新增依赖。重启 FastAPI 并刷新页面后重新生成即可；没有自动重跑旧聊天。

离线验证（临时数据库、禁用真实模型调用，不读写个人记忆）：

```bash
python scripts/run_tests.py
node tests/test_frontend_memory.cjs
node tests/test_frontend_quality.cjs
```

## v0.22 MCP 天气工具

使用官方 MCP Python SDK，新增独立的 `chat_reply_agent.mcp_server`（stdio 服务）及 `get_weather` 只读工具。**不是把 HTTP 函数换个名字叫 MCP**：应用通过 `initialize → tools/list → tools/call` 连接本地服务，由服务查询 Open-Meteo；强模型发出原生 tool call，收到 `role=tool` 的查询结果后继续生成最终回复。

### 工作方式

- 普通聊天仍是一轮强模型请求，不启动 MCP 子进程。不另外调用“判断是否需要天气”的模型。
- 需要天气时由模型决定调用，支持今天、明天或未来7天内的具体日期，以及全天、上午、下午、晚间；日期按查询城市时区计算。
- 提供温度、体感、湿度、降水概率、风速、阵风，以及 PM2.5、PM10、**美国标准 AQI**。天气、空气质量分别查询，空气质量不可用不影响已有天气数据。
- 用于穿衣和出行建议的数值必须来自对应城市、日期、时段；不自动定位，不默认北京/上海，不拿历史居住地当当前所在地。同名城市有歧义会返回候选，应询问用户或根据本轮已有省州信息消歧。
- 雾的天气代码不等于雾霾，湿度高不等于空气污染。工具提供模型估计/预报，不能声称实测、官方预警，也不提供医疗判断。
- 每次生成最多执行2次工具调用、最多3轮强模型请求。一般天气查询为“模型请求工具 → 查询 → 模型生成”两轮；复核仍按需进行。单次 MCP 查询默认18秒上限，成功/部分成功结果缓存5分钟，跨当地日期失效。
- 查询失败、参数错误、缺失数据均作为明确状态回传模型。失败不伪装成成功，缺项不填0；需要支持工具调用的模型。不支持时可暂时设置 `WEATHER_TOOL_ENABLED=false`，不会静默切换成猜天气。
- 天气证据会传给定向复核，避免复核时丢失依据；单条重生成也支持重新查询。工具原始结果不直接存入联系人长期记忆或本人风格。页面“天气依据”显示来源、地点、时段、获取时间和缺项。

可在“聊天背景”中试用：`对方现在在北京，问明晚出去散步穿什么，我想自然地提醒一下。` 不需要额外点查询按钮。是否调用由当前模型决定，端到端回复效果仍需用你的实际模型验证。

### 安装与运行

本次新增 `mcp==1.26.0`，它要求 Pydantic 2.11+，因此不再锁定2.10.4。建议先确认 `my_demo1` 未被其他项目共用；共用环境可给本项目单独建环境，避免互相影响。本次开发验证使用临时隔离环境，没有升级你的 `my_demo1` 或修改真实 `.env`。

```bash
cd /Users/jiaqihong/Desktop/QY_Chat
conda activate my_demo1
# 使用加密依赖预编译包，避免本机缺 OpenSSL/Rust 编译环境
python -m pip install --only-binary=cryptography -r requirements.txt
python -m pip check
python -m uvicorn chat_reply_agent.main:app --host 127.0.0.1 --port 8000
```

默认启用，无需天气 API Key。以下是可选 `.env` 配置，已有大模型配置继续沿用：

```dotenv
WEATHER_TOOL_ENABLED=true
WEATHER_TOOL_TIMEOUT_SECONDS=18
WEATHER_CACHE_SECONDS=300
```

强模型必须支持 OpenAI 兼容的 Chat Completions 工具调用，包括返回 `tool_calls` 以及接收 `role=tool`。不支持时，关闭开关可恢复不带工具的生成。天气工具启动不依赖付费模型密钥。

### 独立 MCP 接入

应用按需自行启动 MCP 服务，无需再开一个终端常驻。其他 MCP 客户端也可配置：

```json
{
  "mcpServers": {
    "chat-reply-weather": {
      "command": "/opt/anaconda3/envs/my_demo1/bin/python",
      "args": ["-m", "chat_reply_agent.mcp_server"],
      "cwd": "/Users/jiaqihong/Desktop/QY_Chat"
    }
  }
}
```

客户端若不支持 `cwd`，使用 `env: {"PYTHONPATH": "/Users/jiaqihong/Desktop/QY_Chat"}`。这些路径需按实际安装位置调整。调用参数示例：`{"query":{"city":"北京","country_code":"CN","region":"北京","day":"tomorrow","period":"evening"}}`。

只注册内置天气工具，不接入任意远程 MCP 地址、命令执行或聊天库读取。子进程不继承大模型密钥；Open-Meteo 只接收查询地点、时段和气象变量，不接收聊天正文、图片、提示词。没有公开网络监听端口。

### 验证与来源

```bash
# 离线测试，含真实 stdio 协议测试（使用无网络的天气替身服务）
python scripts/run_tests.py
node tests/test_frontend_memory.cjs
# 真实天气网络查询，不调用大模型，不产生大模型 API 费用
python scripts/check_weather.py 北京 --country CN --region 北京 --day tomorrow --period evening
```

使用 [MCP 官方 Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.26.0)、[Open-Meteo 天气 API](https://open-meteo.com/en/docs)、[地理编码 API](https://open-meteo.com/en/docs/geocoding-api) 和 [空气质量 API](https://open-meteo.com/en/docs/air-quality-api)。空气质量来源为 CAMS，地点来源为 GeoNames；页面保留署名链接。免费公共端点用于非商业用途，商业部署前应核对 [Open-Meteo 使用条款](https://open-meteo.com/en/terms)。

以下各历史版本说明以本节最新行为为准。

## v0.21 多模型分工

保持 FastAPI、SQLite 和现有关系规则，不引入多 Agent 框架或新依赖。先减少重复模型调用，再用独立角色配置分配模型能力。

### 生成链路

1. 视觉模型只负责识图；保留人工校对。纯文字背景不调用视觉模型。
2. 读取当前联系人的历史和本人风格；**强模型一次返回分析、场景、简短策略及三条候选**。回复时机仍根据分析和场景在本地计算，不增加模型调用。
3. 本地检查后按需定向复核：冲突/道歉/边界场景、中高风险、场景低置信度、说话人不明、立场冲突、重复候选、明显套话或长度超限。最多一批复核，只允许修改命中的候选，不能换掉其他候选。单条重生成仅检查所选候选。
4. 立即返回候选，将记忆任务入队；后台模型整理联系人记忆，再原子保存概要、历史片段及本人风格。

因此普通生成阶段从过去约 5 次串行文本调用变为 **1 次强模型调用，必要时再加 1 次定向复核**。视觉调用次数保持不变，后台记忆另外调用一次。这里减少的是前台等待链路，不是承诺固定秒数；具体延迟仍受模型、输入长度、网络和历史检索影响。

未经模型复核的候选显示“未评分”，不是“通过”。部分复核不计算整组均分；失败保留原文并提示，不会补默认高分。理解与生成请求启用接口级结构化输出，再进行本地严格校验；失败不创建记忆任务，也不把模板包装成真实分析。没有配置模型时仍可看到明确标记的备用示例，但不会把示例分析写入长期记忆。

`MODEL_STRUCTURED_OUTPUT=auto` 默认对百炼 Qwen3.8-Max/Flash、Qwen3.7-Max/Plus/Flash（含快照型号）发送 JSON Schema，约束必填字段、类型、三条候选和场景枚举；其他兼容接口使用 JSON Object 模式并保留本地校验。可显式设置 `json_schema`、`json_object` 或 `prompt`；最后一种只适用于不支持结构化输出的接口，格式稳定性会降低。配置只作用于当前“理解与生成”调用，天气工具往返中保留相同输出约束，不增加模型请求或自动重试，也不改变已选模型与思考模式。支持范围参考[百炼结构化输出文档](https://help.aliyun.com/zh/model-studio/qwen-structured-output)。

生成失败时，`generation_invalid` 日志记录请求 ID、字段路径和错误类型，不记录聊天、模型正文或密钥。空响应、输出截断和模型拒绝分别提示，便于区分接口成功与业务结果有效。修改代码或配置后需停止并重新启动现有服务。

### 模型配置

在实际的 `.env` 中可选增加以下配置（`.env.example` 只是模板，不自动生效）：

```dotenv
# 理解、生成及单条重生；空值继承 OPENAI_MODEL_NAME
OPENAI_STRONG_MODEL_NAME=
# 定向复核及离线评测评分；空值继承强模型
OPENAI_REVIEW_MODEL_NAME=
# 后台联系人记忆整理；空值继承 OPENAI_MODEL_NAME
OPENAI_MEMORY_MODEL_NAME=
# 单次 SDK 网络超时配置，默认 90 秒；关闭隐式自动重试
MODEL_TIMEOUT_SECONDS=90
# 理解与生成的输出约束；默认 auto，无需补写到已有 .env
MODEL_STRUCTURED_OUTPUT=auto
```

识图沿用 `OPENAI_VISION_MODEL_NAME`。四种角色共享现有 `OPENAI_API_KEY`、`OPENAI_BASE_URL`，需填写该服务商实际可调用的模型 ID。不会自动替你选模型、降级或修改真实 `.env`。不设置新增变量也能继续运行，但这时角色可能仍使用同一个模型；配置不同 ID 后才是实际多模型协作。当前不支持每个角色使用不同服务商密钥/地址。

### 后台记忆

- 首次启动自动增加 `memory_jobs` 表，不删除、重建现有数据。建议更新前备份 SQLite 文件。
- 返回 `memory_job_id`、`memory_status=pending`、`memory_persisted=false`；排队并不意味着已经保存完成。页面轮询状态，不刷新或覆盖用户正在编辑的候选。
- 状态为 `pending / running / completed / failed / cancelled`；不保存时为 `disabled`。成功后提供 `saved_memory_id`，解锁反馈操作。生成时参考的画像与联系人记忆保留原快照，避免把事后记忆误认为生成依据。
- 后台使用独立数据库会话，网络等待期间不持有聊天库写锁。联系人记忆、概要、片段及风格在同一个事务中写入，失败整体回滚，只记录错误类型，不暴露聊天或密钥。
- 队列持久化，程序重启会恢复未完成任务；失败不无限重试，可重新生成。取消、删除联系人、手动编辑联系人或清空对应关系风格会阻止旧任务回写。已经保存的记忆仍需在管理页删除。
- 队列含本轮输入，属于本地敏感数据。完成、失败或取消后清除任务输入，完成状态只保留记录 ID 与耗时，不重复保存聊天概要。临时模式不创建任务。
- 使用一个服务进程。后台通过旁路 SQLite 文件 `*.worker-lock` 独占运行，防止同一数据库被多个服务进程重复处理；该锁不锁住聊天数据库，进程退出后自动释放。正常退出会等待当前后台调用结束。
- 查询：`GET /api/memory-jobs/{job_id}?user_id=...`；取消未完成/失败任务：`DELETE /api/memory-jobs/{job_id}?user_id=...`。仍为本地个人应用，`user_id` 是数据隔离标识，不是登录鉴权，服务不要公开到互联网。

### 运行与验证

```bash
cd /Users/jiaqihong/Desktop/QY_Chat
conda activate my_demo1
python -m uvicorn chat_reply_agent.main:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000`。无需新依赖，重启服务并刷新页面即可；不要同时运行两个服务进程指向同一数据库。

耗时面板新增“理解与生成”“定向检查”“记忆任务排队”，后台处理独立记录 `chat_timing`，也可从任务查询的 `diagnostics` 读取。测试使用隔离的临时数据库及模拟模型，不读取真实聊天或消耗 API 额度；真实速度和回复效果需使用你的模型实测。

安全运行离线测试（脚本强制隔离数据库、上传目录并关闭真实模型调用）：

```bash
python scripts/run_tests.py
# 可选：已有 Node.js 时检查页面脚本和记忆状态逻辑
node tests/test_frontend_memory.cjs
```

上述前端检查不是浏览器截图验收，实际页面仍需在浏览器中试用确认。

评测页复用新版理解生成与定向复核，再单独调用只评分、不改写的评测模型。评分结果必须对应实际候选，不能先改好再给高分；因此评测耗时与普通生成耗时不能直接比较。下方 v0.20 内容保留为历史说明，链路和配置以本节为准。

## v0.20 定位与纠偏

本版不接入 Mem0、不更换模型、不迁移数据库，聚焦耗时定位、模板误替换与失败评分修正。没有增加生成链路中的模型调用次数。

### 耗时记录

- 首页结果的“回复质量”区域可展开“本次操作耗时”，查看分析、场景、上下文、历史检索、生成、质检、记忆保存和风格学习的耗时。
- 完整生成及单条重生成响应增加 `diagnostics`，含随机 `request_id`、`total_ms` 和 `stages`。HTTP 响应另有 `X-Request-ID` 和 `Server-Timing`。
- 文本及视觉模型调用单独计时；SDK 内部的重试耗时包含在本次模型调用内，不拆为独立尝试。多张截图有多条视觉调用记录，可在终端中查看。
- 终端输出 `chat_timing` JSON 日志，包括请求编号、HTTP 状态、步骤名称、毫秒数和异常状态。不记录聊天正文、用户标识、提示词、密钥或异常原文。默认不额外写本地日志文件。
- 嵌套步骤不可直接相加：例如 `model.text` 已包含在 `quality.review` 中，历史检索包含在上下文读取中。响应正文总耗时截止生成完成；HTTP 总耗时截止响应头准备完成，不含浏览器渲染、完整网络传输或人工校对等待。
- 错误请求也记录耗时。未配置模型时模型步骤标记为 `skipped`；步骤 `ok` 只表示函数正常返回，不代表模型质检有效，有效性以质量报告 `status` 为准。

这些记录用于找到真实瓶颈；本版不承诺把一分钟的响应缩短到某个固定值。

### 候选与评分

- 超过 46 字或包含 AI 风格词，不再直接替换成“哈哈确实”等通用短句。原文本交给现有质检模型结合当前聊天做最小改写。
- 缺失、重复或与明确回复意图冲突的候选仍有保底处理，但会返回 `source=fallback` 和 `warnings`，页面明确提示备用示例，不算成功的真实生成。
- 质检异常、未配置模型、无效 JSON、缺少维度、重复下标或无效分数，都不能得到默认 80 分。保留原候选，只显示本地规则提醒。
- `overall_score`、各模型评分维度、`average_score` 允许为 `null`，意为“未评分”，不是零分。`local_naturalness` 单独保留规则检查结果，不能代替模型的相关性和关系判断。
- `status` 区分 `reviewed`、`unavailable`、`invalid_response`、`not_reviewed`；整体报告可能为 `partial`。存在未评分候选时不计算整体均分，也不会标记通过。
- 评分对应质检返回的最终文本，不再对该文本做模板替换后沿用旧分数。单条质检若改成另一条已有候选，会保留原句并撤销该次评分。
- 评测页面和命令行同步显示未评分；缺少质量评分的样本不能通过综合评测。单条重生成或手动修改后，前端不沿用不再对应当前文本的均分。

### 真实聊天对照

打开 `/evals`，可以切换到“v0.20 真实聊天对照”运行当前模型评测，也可点击“查看离线对照”，无需模型即可比较旧规则与修正后的行为。

样例位于 `chat_reply_agent/evals/v020_regressions.yaml`，上下文来自用户提供的 2026-06-16 游戏联机微信截图，保留两条相同文本对应不同说话人的原始顺序。**目前只有这一段真实聊天；问题候选和参考改写均为人工构造，不是模型历史输出，也不是用户已发送回复。**

离线对照覆盖长句误替换、风格词误命中、僵硬关怀三个问题。它验证“内容未被粗暴换掉”，不代表修正后已经获得更高的真实回复质量；后者需要运行模型并人工确认。查看对照不会读取现有聊天数据库，不写入记忆。

```bash
cd /Users/jiaqihong/Desktop/QY_Chat
conda activate my_demo1
# 无模型调用、无 API 费用：输出新旧规则复现对照
python scripts/compare_reply_regressions.py
# 使用 .env 中当前模型运行真实聊天对照，会产生模型调用
python scripts/run_evals.py --dataset chat_reply_agent/evals/v020_regressions.yaml --json
# 启动或重启服务
python -m uvicorn chat_reply_agent.main:app --reload
```

更新不需要安装新依赖或修改 `.env`。新增只读接口 `GET /api/evals/regressions`；样本列表支持 `GET /api/evals/cases?dataset=v020`；评测请求可传 `dataset: "v020"`，默认仍使用核心评测集。

## 功能

- FastAPI 后端
- SQLite 本地数据库
- HTML + JavaScript 前端
- 图片上传
- 多模态模型识别聊天截图，输出结构化消息列表
- 支持人工纠错识别结果，并用修正后的内容重新生成建议
- 纠错结果保存到 SQLite，后续可用于优化记忆和风格
- 恋爱关系拆分为“恋人（已在一起）”和“追求/暧昧中”，回复策略会区分亲密边界
- 支持候选回复反馈：采纳、太 AI、太长、太油、太冷淡、答非所问、不符合关系
- 支持手动修改候选回复后保存为风格样本
- 独立风格样本库：采纳或改写过的回复会进入 `style_examples` 表
- 生成回复时会参考“用户真实会发的表达样本”
- v0.7 结构化场景识别：日常、道歉、安慰、邀约、冲突、拒绝、暧昧、冷淡、工作协作和边界表达
- 场景会实际控制回复目标、回复时机、禁忌、口语示例和候选回复，不只是展示标签
- 场景规则集中维护在 `chat_reply_agent/skills/scenario-classifier/references/scenarios.yaml`
- v0.8 联系人记忆：按 `user_id + contact_id` 隔离保存，不同聊天对象不会共享对话事实
- 联系人记忆包含近期概要、稳定事实、互动偏好、最近话题和上次场景
- 生成前读取当前联系人记忆，生成后合并本轮新信息
- 提供联系人列表和单联系人记忆查询接口
- v0.9 回复质量闸门：检查口语感、上下文相关性、关系匹配、场景匹配和用户风格
- 质量不合格时做最小改写，不改变用户立场，不把短消息扩写成长段
- 页面展示整体评分、单条评分、自动改写状态和剩余问题
- 质量模型失败时保留可用候选和本地规则提醒，不伪造模型评分
- v0.10 将 `skills` 从散装 YAML 调整为 Agent Skills 风格目录：每个 Skill 都有 `SKILL.md` frontmatter 和 `references/` 规则文件
- 关系回复规则增加日常表达约束和更生活化的自然样例，减少“AI 味”和情感咨询腔
- v0.11 细化关系和场景 Skill：增加关系雷达、情商动作、回复动作、措辞规则和坏例改法
- 回复策略会显式读取这些细化规则，让模型先判断当前需要“接住、降温、解释、留台阶、轻推进还是明确安排”
- v0.12 回复评测：内置 12 条匿名基准样本，可从 Web 页面或命令行运行
- 评测按场景命中、回复质量、规则合规和候选差异分别评分，不再只凭主观感受调整 Prompt
- 评测样本可关联本地截图，分别计算文字准确率和说话人准确率
- 评测流程不写入联系人记忆、对话概要、风格样本或反馈数据
- v0.13 截图识别改为两阶段：先识别和人工校对，确认后才分析、生成并更新记忆
- 视觉模型返回气泡横向坐标，程序按坐标和左右位置重新计算说话人，不依赖消息语义猜测
- 每条消息返回文字置信度、说话人置信度、归属来源和复核告警
- 对方最后一条消息由程序从规范化消息列表重新计算，不直接相信模型汇总字段
- 相同文字会保留为独立消息，分别根据各自气泡位置判断说话人
- v0.14 联系人级本地 RAG：每轮确认后的聊天保存为独立记忆片段，生成前按当前话题召回最多 3 条相关历史
- 检索按“相似度 × 时间衰减因子”排序，支持本地语义模型和明确标记的关键词回退，不会跨用户或跨联系人读取
- 当前截图不会被当作历史重复召回；页面会展示本次实际参考的片段、日期和相关度
- 历史只作为事实参考，与当前聊天冲突时始终以当前聊天为准
- 提供联系人记忆片段列表和搜索接口，检索本身不增加模型调用或 API 费用
- v0.15 本次回复控制：可选择明确意图，也可补充一句自己的真实目的
- 可调回复长度、亲密程度和主动程度；三条候选必须保持同一核心立场
- 本地规则会拦截明显的立场冲突，例如用户选择拒绝却生成“好啊”
- 每条候选支持独立重新生成，只替换这一条并重新计算它的质量分
- 单条重新生成只读取已有上下文，不重复写入联系人记忆、对话概要、RAG 片段、风格或反馈数据
- v0.16 临时模式：可以继续参考已有上下文生成，但不保存本轮聊天、纠错、RAG 片段或说话风格
- 上传图片只在截图识别期间临时存在，请求结束后即从 `uploads` 删除，模型失败时也会清理
- 新增“记忆与风格”页面，可修改联系人概要、稳定事实、互动偏好和关系类型
- 历史片段支持逐条删除；删除联系人记忆时会同时清理其 RAG 片段及关联的对话概要
- 支持手动添加或删除真人表达样本，并按关系类型重置风格档案、样本和反馈偏好
- 所有编辑和删除操作都按 `user_id`、联系人或关系类型隔离
- v0.17 多截图上下文：一次上传 1-6 张截图，可在页面上预览、排序和移除
- 后端按确认顺序逐张识别，只去除相邻截图衔接处的重复消息，不全局删除重复文本
- 同一截图内真实重复发送的消息会完整保留；无法可靠判断的单个短句边界也会保留给用户校对
- 识别结果返回来源截图数和衔接去重数，页面会明确展示合并情况
- 批量识别中任意一张失败时，当前请求已保存的全部临时图片都会清理
- v0.18 结构化本人语气学习：只读取 `speaker=me` 的消息，不再从整段聊天文本猜测用户风格
- 当前截图里的本人表达会立即参与本次生成，无需等到下一轮
- 自动画像按 `user_id + relation_type` 隔离，最多累计 12 条去重后的本人原句
- 画像会提取短句长度、句尾标点、语气词、笑声、emoji 和短问句习惯
- 首页展示本轮与累计的真人语气依据，管理页可查看自动画像和本人原句
- 临时模式会使用本轮本人语气，但不会把它写入 SQLite
- v0.19 补充聊天背景：最多输入 2000 字，可在不上传截图时单独生成回复
- 文字背景和截图可以同时使用；发生冲突时，以人工确认后的近期聊天消息为准
- 背景会参与分析、场景判断、联系人级 RAG、回复生成和记忆概要，但不会参与用户风格学习
- 结果页会单独展示本次参考的概括背景，并标明它不是双方逐字原话
- services 分层：截图识别、情绪分析、回复时机、候选回复、记忆、说话风格
- skills：按关系类型维护回复规则，程序优先读取 Skill 包里的 `references/rules.yaml`
- OpenAI 配置集中在 `.env`
- 无 `OPENAI_API_KEY` 时可用本地占位逻辑跑通流程

## 运行命令

如果你已经有 conda 环境 `my_demo1`：

```bash
conda activate my_demo1
pip install -r requirements.txt
test -f .env || cp .env.example .env
uvicorn chat_reply_agent.main:app --reload
```

如果需要新建环境：

```bash
conda create -n my_demo1 python=3.11 -y
conda activate my_demo1
pip install -r requirements.txt
test -f .env || cp .env.example .env
uvicorn chat_reply_agent.main:app --reload
```

启动后访问：

```text
http://127.0.0.1:8000
```

## 环境变量

`.env` 中可配置：

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4.1-mini
OPENAI_VISION_MODEL_NAME=gpt-4.1-mini
DATABASE_URL=sqlite:///./chat_reply_agent.db
UPLOAD_DIR=chat_reply_agent/uploads
```

注意：

- 应用实际读取 `.env`，不是 `.env.example`。
- `.env.example` 建议只作为模板；真实密钥请放在 `.env`。
- `.env` 已经在 `.gitignore` 中排除。

即使不填 `OPENAI_API_KEY` 也能启动并返回占位分析。填入密钥后，截图识别、关系分析、场景分类、回复生成、回复质量检查和联系人记忆整理会尝试调用模型；质量检查或联系人记忆整理失败时会自动使用本地规则，不影响候选回复返回。

## 识别纠错流程

1. 选择 1-6 张微信聊天截图，并确认较早的截图排在前面。
2. 点击“识别并校对”。
3. 在校对页面检查每条消息的文字、说话人和左右位置。
4. 点击“确认内容并生成回复”；如果有修改且允许保存本轮，系统会同时保存纠错样本。
5. 生成后仍可继续修正识别内容，并点击“使用修正重新生成”。

## v0.9 回复质量检查

每条最终候选都会返回以下质量维度：

- `naturalness`：是否像真人微信，是否简短口语化
- `relevance`：是否回应当前聊天重点
- `relation_fit`：是否符合恋人、追求、朋友、家人或同事关系
- `scenario_fit`：是否完成当前场景目标并避开禁忌
- `style_fit`：是否接近用户保存的真实表达样本

每个维度为 1-5 分，系统换算为 0-100 总分。某个维度低于要求时，质量服务会先做最小改写，再对最终文本评分。`quality_service.py` 发生模型调用错误时会返回本地评分，不会让整个回复接口失败。

## v0.10/v0.11 Skills 规范化与细化

`chat_reply_agent/skills` 现在按 Agent Skills 方式组织：

```text
chat_reply_agent/skills/
├── friend-reply/
│   ├── SKILL.md
│   └── references/rules.yaml
├── colleague-reply/
│   ├── SKILL.md
│   └── references/rules.yaml
├── family-reply/
│   ├── SKILL.md
│   └── references/rules.yaml
├── lover-committed-reply/
│   ├── SKILL.md
│   └── references/rules.yaml
├── lover-pursuing-reply/
│   ├── SKILL.md
│   └── references/rules.yaml
└── scenario-classifier/
    ├── SKILL.md
    └── references/scenarios.yaml
```

`SKILL.md` 负责说明这个 Skill 什么时候用、怎么用、不要怎么用；`references/*.yaml` 保存程序可读取的结构化规则。这样既更接近 Agent Skills 的规范，也方便后续把某个关系类型单独打磨、测试或迁移到别的 Agent 客户端。

v0.11 中，每个关系规则重点补充了：

- `context_radar`：根据聊天风险判断要轻松、安抚、降温还是明确边界
- `emotional_intelligence`：把“情商”拆成具体动作，例如先站队、给安全感、留台阶、体面退出
- `response_moves`：给模型可选的回复动作，不直接生成动作名
- `wording_rules`：限制抽象词、长句、咨询腔和过度完整的主谓宾
- `bad_to_better`：把 AI 味句子改成真实微信短句的示例

## v0.12 回复评测

启动应用后访问：

```text
http://127.0.0.1:8000/evals
```

页面默认只选中一条样本。每条样本会依次调用当前配置的分析、场景、回复和质量模型；一次运行很多样本会增加模型调用次数和费用，因此建议先单条验证，再运行完整集合。

命令行运行单条样本：

```bash
conda activate my_demo1
python scripts/run_evals.py --case friend_daily_game
```

运行全部样本并输出完整 JSON：

```bash
python scripts/run_evals.py --json
```

默认评测集位于 `chat_reply_agent/evals/dataset.yaml`。文本样本直接填写人工标注的 `messages`；如果要加入截图识别评测，可在样本中增加相对评测集目录的 `image_path`，此时 `messages` 同时作为正确答案。系统会分别计算：

- `text_accuracy`：文字内容与人工标注的相似度
- `speaker_accuracy`：每条消息的发送者是否正确
- `recognition_accuracy`：文字 70% + 说话人 30% 的综合识别分

运行不调用模型的自动化测试：

```bash
python -m unittest discover -s tests -v
```

当前内置样本是用于跑通评测结构的匿名基线。真正比较模型或 Prompt 前，建议逐步补充 30-50 条经过脱敏的真实聊天样本，并为每条样本维护可接受场景、禁用表达和人工参考回复。

## v0.13 截图识别与生成前校对

首页流程调整为：

```text
上传截图
→ 视觉模型识别文字和气泡位置
→ 程序根据位置计算说话人
→ 页面高亮低置信度消息
→ 用户确认或修改
→ 分析场景并生成回复
→ 更新联系人记忆和用户风格
```

`POST /api/vision/extract` 和 `POST /api/vision/extract-batch` 只返回 `ExtractedChat`，不会调用文本分析模型，也不会写入联系人记忆。当前首页使用批量接口，单图接口继续保留用于兼容已有调用。用户确认后，页面调用 `POST /api/reply/from-chat`；如果用户修改了文字、位置或说话人，则改用 `POST /api/reply/corrected` 并保存纠错样本。

每条识别消息新增：

- `bubble_center_x`：气泡中心在图片宽度中的相对位置，范围 0-1
- `text_confidence`：文字识别置信度
- `speaker_confidence`：说话人归属置信度
- `speaker_source`：`geometry`、`side`、`model` 或 `user_confirmed`
- `needs_review` / `warnings`：是否需要人工复核及原因

旧的 `POST /api/reply` 一步式接口继续保留，方便兼容已有调用；当前 Web 首页默认使用新的两阶段流程。

## v0.14 联系人级本地 RAG

v0.14 不再只把联系人总概要整体交给模型，而是把每轮已确认的聊天保存为一个独立的记忆片段，下一次生成时按当前话题选出最相关的历史：

```text
当前已确认聊天
→ 只分析当前场景
→ 在 SQLite 中筛选同一 user_id + contact_id 的历史片段
→ 按相似度 × 时间衰减因子排序
→ 取最多 3 条并和联系人稳定记忆一起注入回复生成
→ 页面展示本次实际使用的历史来源
→ 保存本轮为新的记忆片段
```

数据保存在新增的 `memory_episodes` 表中，包括联系人、关系类型、话题、场景、单轮概要、原始确认文本和本地检索词。应用启动时会通过 SQLAlchemy 自动创建新表，不修改已有表。

当前采用 SQLite 保存记忆、按请求计算相似度的轻量 RAG。支持本地模型或阿里云百炼 Embedding API：

- `MEMORY_EMBEDDING_PROVIDER=local` 使用本地模型；`dashscope` 使用百炼 API；`openai_compatible` 使用其他兼容接口。均计算归一化向量的余弦相似度。失败策略由 `MEMORY_EMBEDDING_FALLBACK` 决定：`error` 报错，`lexical` 才降级并标记 `lexical_fallback`。
- 时间权重乘在相似度上，不再叠加关系、场景或新近度奖励。场景可决定记忆的衰减速度。
- 选择百炼 API 时，当前检索文本与同一联系人的候选历史话题/摘要会发送至阿里云；SQLite 记忆和排序留在本机。选择本地模型时，需额外安装依赖并预先下载模型。
- 最多读取当前联系人最近 60 条候选，最终只向生成模型提供 3 条，控制 Prompt 长度。
- 历史文本按“不可信参考资料”处理，不执行其中的指令，旧内容和当前聊天冲突时以当前聊天为准。

查看某个联系人的最近记忆片段：

```text
GET /api/contacts/xiaolin/memories?user_id=default
```

按话题检索：

```text
GET /api/contacts/xiaolin/memories?user_id=default&query=周末火锅
```

返回结果包含概要、话题、场景、日期、相关度及评分分解，不通过查询接口返回完整聊天原文。当前仍只在最近 60 条候选内排序，不是全量历史向量索引。

### 记忆相关性与时间衰减

```text
时间衰减因子 = 2 ** (-记忆年龄天数 / 半衰期天数)
相关性分数 = 相似度 × 时间衰减因子
页面相关度 = round(相关性分数 × 100)
```

例如语义相似度为 0.8、半衰期为 30 天时，新记忆得分约 80%，30 天后为 40%，60 天后为 20%。排序使用未四舍五入的分数。年龄从片段 `created_at` 起算，反复读取不会使记忆变新；旧数据库的无时区时间按 UTC 解释，未来时间不会使权重超过 1。

在项目根目录 `.env` 中添加配置，然后重启服务：

```dotenv
MEMORY_DECAY_HALF_LIFE_DAYS=30
MEMORY_DECAY_HALF_LIFE_BY_SCENARIO={"invitation":7,"daily_chat":30}
MEMORY_SIMILARITY_BACKEND=semantic
MEMORY_EMBEDDING_PROVIDER=local
MEMORY_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

`MEMORY_DECAY_HALF_LIFE_BY_SCENARIO` 按**每条历史片段自身的 `scenario_type`** 匹配，未配置的场景使用全局半衰期。上述 7 天、30 天只是示例；越短越快遗忘，越长越重视旧记录。空对象 `{}` 表示统一使用全局值。半衰期必须为有限正数，零、负数、NaN、无穷大会导致配置校验失败。

衰减只影响 `memory_episodes` 的召回排名，不删除历史，不修改联系人稳定事实或用户说话风格。语义模式在衰减后至少 16% 才召回；关键词模式取消旧有加分后使用 10% 门槛。这是检索门槛，不是情绪分析置信度，后续应结合真实聊天评测调整。

#### 使用千问 Embedding API

已有 `.env` 时只补充或修改下面的字段，不要直接用 `.env.example` 覆盖，否则会丢失现有密钥和配置。仅首次安装且不存在 `.env` 时才复制示例文件。

```dotenv
MEMORY_SIMILARITY_BACKEND=semantic
MEMORY_EMBEDDING_PROVIDER=dashscope
MEMORY_EMBEDDING_MODEL=qwen3-vl-embedding
MEMORY_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MEMORY_EMBEDDING_API_KEY=
MEMORY_EMBEDDING_REUSE_CHAT_CREDENTIALS=false
MEMORY_EMBEDDING_FALLBACK=error
MEMORY_EMBEDDING_TIMEOUT_SECONDS=20
```

默认不自动复用聊天地址或密钥。请填写同地域/业务空间的向量地址和密钥。确需复用时显式设置 `MEMORY_EMBEDDING_REUSE_CHAT_CREDENTIALS=true`，且向量与聊天的基础地址必须完全相同；不同地址必须使用独立向量密钥。新业务空间按控制台给出的兼容接口地址配置。`qwen3-vl-embedding` 会在相同主机上转换为原生 API 路径，其他兼容模型仍使用已有 SDK；均无需安装 `requirements-memory.txt` 或下载本地模型。

示例模型为 `qwen3-vl-embedding`，原生接口按每批最多 20 段文字处理，关闭融合模式。对最多 60 条候选加查询重新计算向量时，最多产生 4 次 API 请求。切换为 `text-embedding-v4` 时仍按兼容接口每批最多 10 段处理，最多 7 次请求。所有批次共享超时预算，无自动重试；任何批次或向量校验失败，整次检索按配置报错或降级，不使用部分返回结果。当前不持久化向量，两种接口都会增加网络耗时和用量。

修改后重启服务，`/api/status` 的 `memory_retrieval` 显示已选择的后端、提供方及模型（仅表示配置，不代表接口已连通）。检索结果的 `similarity_method=semantic` 才说明该次实际完成语义计算；`lexical_fallback` 表示已降级。相似度阈值沿用现有值，切换模型后应结合聊天样本重新评估。

官方说明：[百炼多模态向量接口](https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference)、[文本向量兼容接口](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)。

#### 使用本地 Embedding 模型

设置 `MEMORY_EMBEDDING_PROVIDER=local`，并将 `MEMORY_EMBEDDING_MODEL` 设为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 或本机模型目录。本地模式不调用在线 Embedding API；依赖或模型不可用时，时间衰减仍然生效，但会使用关键词匹配并标记降级。启用方式：

```bash
cd /Users/jiaqihong/Desktop/QY_Chat
conda activate my_demo1
python -m pip install -r requirements-memory.txt
python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")'
python -m uvicorn chat_reply_agent.main:app --reload
```

下载命令需要联网且只需预先执行一次；也可以将 `MEMORY_EMBEDDING_MODEL` 配成本机已有模型目录。聊天请求只加载本地缓存，禁止临时联网下载，加载失败后重启可重新尝试。模型在进程内复用，每次将查询和候选概要批量编码，不持久化向量或额外缓存聊天文本；首次加载和本地推理仍有耗时，尚未做真实模型性能评测。相关 API 用法见 [Sentence Transformers 文档](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)。

返回字段 `similarity_score`、`time_decay_factor` 均为 0–1；`decay_half_life_days` 为本条记忆使用的半衰期，`similarity_method` 为 `semantic`、`lexical`、`lexical_fallback` 或 `not_scored`。不带搜索条件的历史列表不计算分数，因此分解字段为 `null`，不会额外运行模型。首页历史来源和记忆管理搜索结果会标明实际匹配方式及时间权重。

## v0.15 回复意图与单条重生成

生成前可以设置：

- `我想怎么回`：让系统判断、答应、拒绝、安慰、解释、主动推进、先降温、结束话题或问清楚
- `补充意图`：用自己的话说明这次想达到的效果，例如“拒绝，但给对方留个台阶”
- `回复长度`：很短、短句或稍完整
- `亲密程度`：克制、自然或更亲近
- `主动程度`：只回应、自然接话或主动推进

这些设置会同时进入回复策略、候选生成和质量检查。用户明确选择的立场优先于模型自行猜测，三条候选只改变措辞和语气，不允许一条答应、一条拒绝。对于答应、拒绝、降温和结束话题等可明确判断的冲突，程序还会在模型输出后做一次本地兜底检查。

生成结果中的每条候选都有“重新生成这条”按钮。它会调用：

```text
POST /api/reply/regenerate
```

接口沿用本轮已经确认的聊天、分析、场景和回复设定，只替换指定的一个候选并避免与另外两条重复。这个操作不会重新分析截图，也不会新增联系人记忆、对话概要、RAG 片段、风格样本或反馈记录；只有完整确认并生成一轮回复时才会更新记忆。

## v0.16 可控记忆与隐私

首页“本次回复设定”中新增“保存本轮到联系人记忆”复选框。取消勾选后：

- 仍会读取当前联系人已经存在的记忆和相关历史，保证回复上下文连贯。
- 不会新增或更新联系人概要、对话概要、RAG 历史片段和自动风格档案。
- 人工修正过的截图识别结果也不会写入纠错表。
- 返回结果中的 `memory_persisted` 为 `false`，`saved_memory_id` 和 `saved_correction_id` 均为空。

无论是否勾选保存，截图文件都只用于本次视觉识别，并在请求结束时删除。为了避免未经确认的破坏性操作，升级前已经留在 `chat_reply_agent/uploads` 中的旧图片不会被自动删除。

临时模式只控制本机 SQLite 和截图临时文件，不会跳过模型调用。截图和确认后的聊天内容仍会发送给 `.env` 中配置的大模型服务；服务商侧的数据处理和保留规则取决于你使用的 API 提供方。

启动应用后访问：

```text
http://127.0.0.1:8000/memory
```

管理页可以完成：

- 编辑联系人备注、关系类型、概要、稳定事实、互动偏好、最近话题和场景。
- 搜索或逐条删除本地 RAG 历史片段。
- 删除某个联系人的概要、全部历史片段及其关联对话概要。
- 手动添加、查看或删除用户真实会发的表达样本。
- 按关系类型重置自动风格档案、风格样本和反馈偏好。

主要管理接口：

```text
PUT    /api/contacts/{contact_id}/memory
DELETE /api/contacts/{contact_id}/memory
DELETE /api/contacts/{contact_id}/memories/{memory_id}
POST   /api/style/examples
DELETE /api/style/examples/{example_id}
DELETE /api/style/data
```

这里的 `user_id` 是本地数据命名空间，不是登录认证。应用默认只监听 `127.0.0.1`；如果后续开放到局域网或公网，需要先增加真正的账号认证和访问控制。

## v0.17 多截图识别与边界合并

首页支持一次选择 1-6 张截图。文件列表中的序号就是识别和合并顺序，可以在提交前上移、下移或移除。后端会按这个顺序逐张调用视觉模型，因此上传 N 张截图会产生 N 次视觉模型调用；建议只选择覆盖当前上下文所需的截图，避免不必要的费用和等待。

合并不会对整段聊天做普通的“文本去重”，而是只比较前一段结尾和后一段开头：

- 发送者和规范化后的文字都一致，才可能视为截图衔接重叠。
- 两条及以上连续消息匹配时，移除后一张截图开头的重复部分。
- 只有一条较长消息匹配，或两边存在相同的明确时间时，才自动移除。
- 单个“好”“哈哈”“其实差不多”之类的短句边界默认保留，避免把对话中真实重复发送的消息误删。
- 同一张截图内部的重复消息永远不会被合并。

`ExtractedChat` 新增两个兼容字段：

- `source_image_count`：本次合并的来源截图数。
- `merged_overlap_count`：在截图衔接处移除的重复消息数。

批量识别接口：

```text
POST /api/vision/extract-batch
multipart/form-data: images=<1-6 个图片文件>
```

批量请求中的图片仍然只作临时识别使用。全部识别成功、模型调用失败或后续合并异常时，当前请求保存的所有临时图片都会在返回前删除。v0.17 没有增加数据库表，也没有新增依赖。

## v0.18 可靠的本人语气学习

旧版自动风格服务接收的是拼接后的整段聊天文本，存在两个问题：可能把对方的语气算到用户画像里，而且旧解析规则只识别 `我:`，无法稳定匹配实际生成的 `我 (right):` 文本。v0.18 改为直接读取经过用户校对的结构化消息：

```text
ExtractedChat.messages
→ 只保留 speaker=me
→ 清理、截断和去重本人原句
→ 提取本人表达特征
→ 当前轮立即注入回复生成和质量检查
→ 勾选保存时累计到对应关系类型的 UserStyleProfile
```

自动画像目前会提取这些可观察特征：

- 本人消息的中位长度，区分很短反应句、微信短句和较完整表达。
- 是否习惯在句尾使用完整标点。
- 常用的语气词，例如“呀”“啦”“呢”“吧”。
- 是否常用“哈哈”或 `hhh` 缓和语气。
- 是否使用 emoji 或短问句自然接话。
- 最多 6 条当前本人原句和 6 条累计本人原句会作为生成依据；只模仿长度、语气词、停顿和标点，不复用旧事实、称呼或承诺。

首页生成结果中的“真人语气依据”会展示本轮识别到多少条本人消息、提取出的特征和实际参考原句。“记忆与风格”页面会展示按关系类型保存的自动画像，并继续单独管理采纳、手动改写和手动添加的风格样本。

自动画像继续使用已有的 `user_style_profiles` 表，不需要数据库迁移。累计原句保存在现有 `example_phrases` 字段中，最多保留 12 条最近且不重复的本人表达。关闭“保存本轮到联系人记忆”后，本轮语气仍会用于当前回复，但不会更新画像。

`.env.example` 只保留空的 `OPENAI_API_KEY` 模板，真实密钥应只写在已被 `.gitignore` 排除的 `.env` 中。如果密钥曾经写入模板并提交到远端或分享给其他人，应在对应模型平台轮换密钥。

## v0.19 文字背景与截图联合上下文

首页新增“补充聊天背景”输入框，适合两种情况：暂时不方便上传聊天截图，或更早的聊天记录太长、不适合全部截图。输入框最多接收 2000 个字符，截图和文字背景至少提供一种。

三种使用方式：

- 只上传截图：保持原有流程，先识别并校对，再生成回复。
- 只填写背景：跳过视觉识别，直接分析概括背景并生成回复。
- 背景加截图：先把背景作为关系来龙去脉，再把校对后的截图消息作为更近、更可靠的聊天证据。

发送给文本模型前，程序会构造带有明确边界的上下文：

```text
上下文使用规则
用户补充背景（概括，非逐字聊天）
近期聊天（截图识别并由用户确认）
```

背景和近期聊天都被视为待分析数据，而不是可执行指令。若两者存在矛盾，Prompt 会要求模型以近期聊天为准；模型也不能把背景概括伪装成对方说过的原话。

补充背景会参与情绪与态度分析、场景分类、联系人级历史检索、候选生成、质量检查和本地记忆。自动说话风格仍然只读取 `ExtractedChat.messages` 中 `speaker=me` 的消息，因此背景里即使出现“我平时会说……”之类文字，也不会被当作本人原句保存。仅文字模式使用 `source_image_count=0` 表示本轮没有截图。

请求字段示例：

```json
{
  "relation_type": "friend",
  "background_context": "我们昨天因为见面时间吵了几句，今天对方主动联系。",
  "extracted_chat": {
    "messages": [],
    "source_image_count": 0
  },
  "save_to_memory": true
}
```

v0.19 沿用现有 SQLite 表保存组合后的上下文，不需要数据库迁移，也没有新增 Python 依赖。

## v0.8 联系人记忆

页面中的联系人字段：

- `聊天对象 ID`：稳定标识同一个人，例如 `xiaolin`，用于隔离记忆。
- `对方备注`：页面展示名称，例如“小林”，后续可以修改。
- 同一用户再次输入已有联系人 ID 时，系统会自动读取该联系人的历史记忆。

联系人记忆保存在新增的 `contact_memories` 表中。启动应用时会自动创建新表，不修改已有表结构，也不需要手工迁移旧数据库。

查看当前用户的联系人列表：

```text
http://127.0.0.1:8000/api/contacts?user_id=default
```

查看单个联系人记忆：

```text
http://127.0.0.1:8000/api/contacts/xiaolin/memory?user_id=default
```

## 场景识别与风格反馈

每次生成会先得到标准化的 `scenario` 对象，包括：

- `type` / `label`：场景代码与中文名称
- `confidence`：场景判断置信度
- `signals`：模型依据的聊天信号
- `reply_goal`：该场景下首先要完成的回复目标
- `cautions`：需要避开的回复方式

场景结果会传给 `timing_service.py` 和 `reply_service.py`。例如冲突场景优先降温，拒绝场景避免纠缠，安慰场景先接住情绪，邀约场景要求给出明确安排。

反馈和风格样本流程：

1. 查看 3 条候选回复。
2. 如果某条大致合适，可以先在文本框里改成自己的说法。
3. 点击“采纳”保存为正向样本，系统会写入风格样本库。
4. 如果不满意，点击“太 AI”“太长”“太油”“太冷淡”或“不符合关系”。
5. 如果你对不满意的回复做了手动改写，改写后的文本也会写入风格样本库。
6. 下一次生成回复时，系统会读取这些反馈偏好和真实表达样本。

查看风格样本接口：

```text
http://127.0.0.1:8000/api/style/examples?user_id=default&relation_type=lover_pursuing
```

## 配置检查点

打开首页后，页面顶部会显示模型配置状态：

- `已配置模型`：说明应用读到了 `.env` 中的 `OPENAI_API_KEY`。
- `未配置 .env`：说明当前还在使用本地占位识别。

也可以访问：

```text
http://127.0.0.1:8000/api/status
```

如果提交图片后报模型调用失败，优先检查：

- `.env` 是否存在
- `OPENAI_API_KEY` 是否填写
- `OPENAI_BASE_URL` 是否正确
- `OPENAI_VISION_MODEL_NAME` 是否支持图片输入
- API 额度和网络是否可用
