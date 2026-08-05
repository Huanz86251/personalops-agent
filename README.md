<div align="center">

# PersonalOps Agent

一个运行在飞书里的个人任务 Agent。

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Feishu](https://img.shields.io/badge/Feishu-Personal_Agent-3370FF?logo=lark&logoColor=white)](https://open.feishu.cn/)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-LLM-4D6BFE)](https://www.deepseek.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/MCP-Playwright-2EAD33?logo=playwright&logoColor=white)](https://modelcontextprotocol.io/)

这是一个给个人使用的 Agent 项目，目前没有支持多用户。现阶段主要解决几个工程问题：复杂任务怎么分步执行、模型一次能看到哪些工具、长对话怎么压缩，以及程序中断后如何继续之前的状态。

</div>

---

## 👤 项目定位

PersonalOps Agent 通过飞书长连接接收消息，并使用 DeepSeek 完成对话理解、任务拆分与结果生成。

对于简单问题，系统可以直接回答；对于需要检索、浏览器操作或文件处理的任务，系统会先判断是否需要规划，再按步骤调用对应工具，最后统一检查结果。

这个项目目前更接近一套个人 Agent 实验实现，而不是开箱即用的 SaaS 产品。代码重点放在运行链路、上下文管理、工具边界和可恢复状态上。

> **面向个人使用：** 当前版本只考虑一个使用者。不同飞书对话共享同一套用户记忆，但每个对话仍有独立的聊天记录；同一对话中的消息会按顺序处理，避免多个任务同时修改同一份状态。

<br>

## ✨ 已实现的能力

### 飞书对话入口

- 使用飞书应用长连接接收消息，无需额外提供公网回调地址
- 单用户消息串行处理，避免连续消息覆盖同一轮状态
- 支持创建、查看和切换独立对话
- 任务执行过程中可回传阶段性进度

可用命令：

```text
/help               查看使用说明
/new [标题]         创建并切换到新对话
/list               查看最近对话
/switch 编号或短ID  切换对话
/current            查看当前对话
```

### 分层任务执行

复杂任务不会让一个 Agent 无限制地连续调用工具，而是按下面几个阶段执行：

1. **Hard Supervisor** 判断请求可以直接完成，还是需要进入计划执行。
2. **Planning Graph** 生成执行计划，同时限制最多能调用多少次模型和工具，以及失败后允许重试几次。
3. **Simple Executor** 只处理当前步骤，并按需选择工具集。
4. **Step Reporter** 总结这一步调用了哪些工具、得到什么结果，以及是否已经完成。
5. **Hard Final Reviewer** 检查最终结果。发现任务还缺关键步骤时，最多允许重新规划一次。

模型每次只会看到当前任务可能用到的工具，而不是一次加载所有工具。

### 动态工具集

项目按照能力域组织工具，而不是把全部工具长期暴露给模型。

当前注册的 Toolset 包括：

| Toolset | 用途 |
| --- | --- |
| `WEB_RESEARCH` | 时间确认、网页搜索和公开信息读取 |
| `BROWSER_AUTOMATION` | 页面导航、点击、输入和表单交互 |
| `FILE_INSPECTION` | 目录浏览、文件查找、文本读取与搜索 |
| `FILE_EDITING` | 文本文件创建、写入和精确替换 |
| `SOFTWARE_DEVELOPMENT` | Python、Git、测试及其他 Shell 调试任务 |

Toolset Router 会先选择当前步骤需要的工具集。如果执行过程中发现工具不够，Executor 可以通过 `request_toolset` 申请其他工具。启动时没有成功注册的工具不会提供给模型。

### Playwright MCP

浏览器能力通过 Playwright MCP 接入。

程序启动时会连接 Playwright MCP，并只注册白名单中的浏览器工具。目前支持页面导航、快照、点击、输入和多标签页操作。Agent 只负责选择工具，具体的浏览器操作由 MCP 服务执行。

### 多会话与可恢复状态

- LangGraph Checkpoint 使用 SQLite 持久化
- 每个 Conversation 使用独立 thread id
- 对话可通过列表编号或短 ID 切换
- 同一对话中的任务执行和记忆写入会串行完成，避免并发修改造成状态冲突
- 退出时依次停止消息接收、等待任务结束，再关闭 SQLite、Memory Store 与 MCP 资源

运行数据默认保存在 `.agent/`，该目录不会提交到 Git。

<br>

## 🧩 上下文是如何控制的

为了避免对话和工具输出越来越长，项目在两个地方做了摘要：一处压缩较早的聊天记录，另一处压缩单个执行步骤里的工具调用历史。

### Conversation Rolling Summary

当对话累计到配置阈值后，较早的对话会被更新为滚动摘要。Hard Supervisor、Replanner 和 Final Reviewer 使用这份摘要，同时只携带有限数量的最近对话。

摘要主要保留用户目标、已经确认的限制和前面完成的结果，因此后续规划不需要每次重新读取完整对话。

### Executor Trace Summary

每个执行步骤也有独立的上下文上限。当模型回复和工具结果过长时，系统会压缩较早的内容，只保留摘要和最近几条消息继续执行。

如果程序恰好在工具调用过程中中断，Middleware 会先补齐缺失的 AI/Tool 消息关系，再进行摘要，避免恢复后消息顺序不合法。模型摘要失败时，会改用本地文本截取和拼接逻辑继续运行。

<br>

## 🧠 记忆检索

长期记忆保存在 SQLite Store 中，并使用本地模型完成检索：

- `gte-multilingual-base`：向量表示
- `gte-multilingual-reranker-base`：候选重排
- `Qwen3 GGUF`：轻量记忆路由判断，可通过配置关闭

检索到一条记忆后，系统还可以沿着内存关系图找到与它关联的其他记忆。例如命中某个任务后，可以继续找到相关联系人、文档或历史对话。关系图只用于查询，最终数据仍以 SQLite 中的记录为准

模型文件默认下载到 `.models/`，不会提交到仓库。

<br>

## 🔄 运行链路

```mermaid
flowchart TD
    A["飞书消息"] --> B["Conversation Runtime"]
    B --> C["读取 Checkpoint、摘要与相关记忆"]
    C --> D["Hard Supervisor"]
    D -->|直接完成| H["返回结果"]
    D -->|需要执行| E["Planning Graph"]
    E --> F["Simple Executor + 动态 Toolset"]
    F --> G["Step Reporter"]
    G --> I["Hard Final Reviewer"]
    I -->|完成| H
    I -->|需要调整且预算允许| J["Hard Replanner"]
    J --> E
    F -.-> K["Playwright MCP / 文件 / 搜索 / Shell"]
    B -.-> L["SQLite Checkpoint 与长期记忆"]
```

<br>

## 🚀 本地运行

### 1. 准备环境

项目开发环境为 Python 3.10。仓库同时提供 Conda 和 pip 依赖文件。

```powershell
git clone https://github.com/Huanz86251/personalops-agent.git
cd personalops-agent
conda env create -f environment.yml
conda activate agent
```

首次启动记忆模块时会下载本地模型，因此需要预留磁盘空间并保持网络可用。

浏览器 MCP 通过 `npx` 启动，还需要本机已安装 Node.js。

### 2. 配置环境变量

复制示例文件：

```powershell
Copy-Item .env.example .env
```

至少填写以下配置：

```dotenv
FEISHU_APP_ID=
FEISHU_APP_SECRET=
DEEPSEEK_API_KEY=

LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
HARD_LLM_PROVIDER=deepseek
HARD_LLM_MODEL=deepseek-v4-pro
```

其余规划预算、记忆模型和观测配置可以先沿用 `.env.example` 中的值，再根据本机资源调整。

### 3. 配置飞书应用

在飞书开放平台创建企业自建应用，并为应用配置机器人与消息接收能力。项目使用长连接接收事件，请确保应用侧已经启用相应事件订阅与权限。

不同飞书应用的权限范围可能不同，具体配置以飞书开放平台当前说明为准。

### 4. 启动

```powershell
python main.py
```

启动完成后，终端会保持运行并等待飞书消息。

如果启用了 Phoenix Tracing，可在默认地址查看本地调用轨迹：

```text
http://127.0.0.1:6006
```

Phoenix 启动失败不会阻止飞书入口和 Agent 主流程继续启动。

<br>

## 🗂️ 目录说明

```text
.
├─ main.py                    # 飞书入口与应用生命周期
├─ conversation_runtime.py    # 会话、Checkpoint、Memory 与 MCP 编排
├─ planning_graph.py          # 计划执行状态图
├─ hard_planning.py           # Supervisor、Replanner、Final Reviewer
├─ agent.py                   # Executor Agent 与模型构造
├─ middlewares.py             # 执行预算与 Toolset 路由
├─ context_middlewares.py     # 消息修复、摘要与运行时 Prompt
├─ memory.py                  # 长期记忆提取、解析和检索
├─ memory_graph.py            # 记忆关系图索引
├─ retrieval_models.py        # 本地向量、重排和路由模型
├─ mcp_runtime.py             # Playwright MCP 生命周期
├─ observability.py           # Phoenix / OpenTelemetry 观测
├─ toolsets.py                # Toolset 定义与能力目录
├─ tools/                     # 文件、搜索、时间等基础工具
└─ prompts/                   # 各节点使用的 Prompt
```

运行时目录：

| 路径 | 内容 |
| --- | --- |
| `.agent/` | Checkpoint、长期记忆、Phoenix 数据与浏览器状态 |
| `.models/` | 本地模型缓存 |
| `workspace/` | Agent 文件工具可操作的工作区 |

这些目录已通过 `.gitignore` 排除。

<br>

## 🧭 当前边界

- 当前只支持单用户，不包含账号、团队或权限隔离功能
- 主要面向 Windows 与本机常驻运行方式
- 浏览器自动化依赖本地 Node.js 和 Playwright MCP
- 本地检索模型会带来额外的磁盘与内存占用
- 网页或 Shell 工具可能因为网络、网站限制或本机权限失败
- 仓库目前以实现代码为主，自动化测试覆盖仍需继续补充
- 目前选用的搜索引擎实际体验略差，广告较多，或页面经常需登录



---

<div align="center">

Built as a personal agent runtime for Feishu.

</div>
