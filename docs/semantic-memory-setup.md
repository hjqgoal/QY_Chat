# 语义向量 RAG 记忆检索

## 原来为什么没有用上

项目本来就有 RAG 流程：从 SQLite 查找相关记忆，再把找到的历史加入回复模型的上下文。RAG 不一定要用向量；原先本机的真实 `.env` 没有 `MEMORY_EMBEDDING_*` 配置，程序默认选择本地 Sentence Transformers，但依赖未安装，因此用的是词面检索。

本次补齐了 API 向量配置、独立密钥、严格模式、每次检索状态和合成用例自检。无需安装本地 embedding 依赖，也不需要额外部署向量数据库。**代码已接通不等于你的 API 已验证；填好配置并完成下面自检后，才能确认真实语义链路可用。**

## 你需要填写的内容

编辑项目根目录真实 `.env`，不是 `.env.example`。本次已添加下列配置块，保留了原有聊天模型配置；不要再次重复追加相同变量。

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

必须确认三项：

1. `MEMORY_EMBEDDING_API_KEY`：在本机填写有向量模型权限的密钥，不要发到聊天里或提交 Git。
2. `MEMORY_EMBEDDING_MODEL`：当前使用你选择的 `qwen3-vl-embedding`。它是多模态向量模型，不是回复生成模型；本项目目前只向它发送文字查询和记忆概要。
3. `MEMORY_EMBEDDING_BASE_URL`：填写与你的密钥地域/业务空间匹配的兼容接口**基础地址**，不要附加 `/embeddings`。已预填与现有聊天配置相同的旧版百炼基础地址；是否适用于你的业务空间须以控制台为准。新空间可能使用 `https://实际业务空间ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`，请勿保留占位符。

### qwen3-vl-embedding 原生接口（2026-10-23）

`MEMORY_EMBEDDING_PROVIDER=dashscope` 且模型为 `qwen3-vl-embedding` 时，程序自动使用百炼原生多模态向量 API，不再通过 `client.embeddings.create` 请求这个模型。无需修改现有 `.env` 的兼容基础地址，也无需更换密钥。

- 基础地址接受 `/compatible-mode/v1` 或 `/api/v1` 结尾。请求时只将路径替换为 `/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding`，保持你配置的主机、地域和业务空间不变，不自动重定向。
- 输入使用 `input.contents=[{"text": "查询"}, {"text": "概要"}, ...]`，显式设置 `parameters.enable_fusion=false`，让每段文字各自生成向量。不能把查询和所有记忆融合成一个向量。
- 每批最多 20 段文字，查询占一段。读取 `output.embeddings`，按返回的 `index` 排序，验证数量、独立向量类型、维度及数值。
- 这个模型走原生接口，不代表项目已实现截图向量存储或图文混合检索。当前仍是文字记忆 RAG。
- 如果返回 401/403/404/429，页面会显示 HTTP 状态及对应检查方向，不输出上游可能包含密钥或聊天内容的报错正文。

[百炼多模态向量官方接口](https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference)

原来的 `text-embedding-v4` 仍可使用，设为该名称后继续走兼容文本向量接口。[文本向量官方接口](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)

使用其他兼容 `/embeddings` 的 HTTPS 服务时，设 `MEMORY_EMBEDDING_PROVIDER=openai_compatible`，并单独填写它的模型、地址和密钥。兼容协议不代表所有厂商的模型、参数和限制都一致，需运行自检。

默认**不复用聊天密钥**。若确认向量与聊天使用同一个地址和密钥，可设 `MEMORY_EMBEDDING_REUSE_CHAT_CREDENTIALS=true`，向量密钥留空；只有完整基础地址相同才允许复用。不同地址必须独立配置，防止把聊天密钥误发到另一个服务。

## 严格模式

- `MEMORY_EMBEDDING_FALLBACK=error`：检索历史需要向量时，配置缺失、超时、返回非法向量等情况明确报错；不调用后续回复模型，不把关键词结果冒充语义结果。
- `MEMORY_EMBEDDING_FALLBACK=lexical`：允许降级保住可用性，页面和 API 会标为 `lexical_fallback`，但不能把这次运行称为语义检索。
- 没有历史记录时不调用向量 API，标为 `not_scored`，仍可生成第一轮回复。保存完成后，后续查询才有历史可以召回。
- 仅打开网页/查看配置状态不调用 embedding，也不产生向量 API 费用。状态显示“已配置，尚未验证”，不会直接标“已连通”。

## 验证与启动

```bash
cd /Users/jiaqihong/Desktop/QY_Chat
conda activate my_demo1

# 仅检查配置，不调用 API
python scripts/check_embedding_connection.py

# 填好配置后运行：只发送合成查询和两条合成摘要，可能产生少量向量费用
python scripts/check_embedding_connection.py --live

# 停止原服务后重新启动，并刷新页面
python -m uvicorn chat_reply_agent.main:app --host 127.0.0.1 --port 8000
```

`--live` 使用临时的内存 SQLite，不读取个人聊天、不发送截图、不调用回复生成模型。它验证：

1. 配置的向量 API 返回有效且维度一致的数值向量。
2. “晚饭想吃点不刺激的”能把“偏好清淡、吃辣不舒服”的记忆排在“周末游戏”之前。
3. 其他联系人的记忆不参加该次检索。
4. 召回的摘要确实进入生成上下文的构造结果。

成功时输出 `semantic_rag_verified: true` 和 `retrieval.method: semantic`，失败则非零退出；配置检查没有 `--live` 时不声称验收通过。该自检只覆盖一个合成场景，不代表真实场景召回准确率。离线测试另行验证强模型提示实际收到召回内容，真实自检不会为此多付一次聊天模型费用。

应用中验证：先用一个合成联系人保存一轮“清淡饮食偏好”，等待记忆任务完成，再以“晚饭想吃点不刺激的”生成回复，检查“本次参考的历史”。也可以在“管理记忆与风格”搜索该联系人历史。

API 的 `retrieval.method` 表示本次实际方法，即使没有召回结果也会返回：

| 值 | 含义 |
| --- | --- |
| `semantic` | 已成功计算向量并按相似度/时间衰减检索 |
| `lexical_fallback` | 向量不可用，本次降级了 |
| `lexical` | 主动配置为词面检索 |
| `not_scored` | 没有历史、没有查询，或只是读取历史列表；未做相似度检索 |

状态接口 `/api/status` 只检查配置，不保留或假装记住自检成功。运行时是否成功以本次响应为准。

## 当前实现与边界

```text
当前聊天/补充背景 -> 查询文本
SQLite -> 当前 user_id + contact_id 的最近 60 条候选摘要
Embedding API -> 查询向量与候选向量
归一化余弦相似度 × 2 ** (-年龄天数 / 半衰期天数)
相关性阈值筛选 -> 默认最多 3 条记忆 -> 强模型回复上下文
```

查询和历史必须用同一服务、同一模型进行编码。程序校验批次条数、下标、维度、有限数值和非零范数；不把半截批次当完整结果。现有时间衰减与联系人隔离保持不变。

SQLite 仍持久化文本记忆，**当前向量在请求内计算，不持久化**。这仍是向量检索 RAG，不需要先装向量数据库；但不是全量历史 ANN 向量索引。`qwen3-vl-embedding` 每批最多 20 段，最多 60 条候选加查询会有最多 4 个请求；原兼容接口按每批 10 段处理，最多 7 个请求。两者都会增加延迟和费用。所有批次共享超时预算，无自动重试；后续优化可以为摘要缓存向量、做后台索引，不能宣称本次已经做到。

API 模式会将当前查询和候选摘要发送给你配置的服务商。向量不是隐私匿名化措施。仍应只使用获得授权的聊天资料；服务商数据政策与你本地 SQLite 存储是两件事。

本次接口适配不更换聊天/视觉模型，不修改个人记忆，不需要数据库迁移或新增 Python 依赖。保留你在真实 `.env` 填写的密钥；`.env.example` 只提供示例，不保存真实密钥。修改配置或更新代码后重启服务，自检成功后再对外展示“真实语义向量 RAG 已验证”。

工程验证：本次 243 项后端离线测试通过，其中新增 12 项原生向量接口测试，覆盖分批、乱序、异常返回、超时、密钥保护、禁止重定向和 RAG 上下文传递。上一轮已通过 4 组前端检查与 1280px/390px 浏览器回归，本次不修改前端。模拟接口测试不替代真实 `--live` 自检。
