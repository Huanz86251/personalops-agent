# 飞书附件接收：Checkpoint 1

后续 General 已改为[自行检查并直接汇报的简单流程](general-simple-flow.md)，不再经过独立 Step Reporter。以下附件接收、任务绑定和共享路径约定继续有效。

2026-09-05。用户仅批准这一阶段：接收图片/文档、保存、关联任务并接通已有阅读工具。
未实现文件发送、桌面文件传回、邮件或本地应用控制；没有重启正在运行的 Agent。

## 使用方式

- 单独发送图片或 PDF 等文件：先保存并返回 `att_...` 编号，不自动猜测用户任务。
- 回复**用户原先发送的附件消息**并写要求，或在新指令中写该编号，例如
  `读取 att_...，提取金额`。暂不通过回复机器人的确认消息来绑定附件。
- 飞书富文本中的图片＋文字、SDK 合并消息中的附件＋文字：一同进入任务。
- `/insert 要求`、`/replace 要求` 可以引用附件；纯 `/cancel`、`/new` 等控制命令
  绕过附件下载队列，继续走原有命令入口。
- 引用限定于同一飞书发送者、聊天和 PersonalOps Conversation。切换 Conversation
  后需要重新上传；不会把未指定的“最近一个文件”自动交给任务。

## 实际链路

```text
Feishu SDK InboundMessage.resources / batched_sources
  → 下载消息资源（file_key + 原始 message_id + resource_type）
  → 按字节识别格式；保存原文件、SHA-256、发送者/聊天/会话/消息关联
  → 单文件消息：返回编号，等待明确指令
  → 有指令：持久化附件与 event_id 的关联，再写入既有 Event Store
  → EventRunPump 执行或恢复：验证文件校验和，建立该任务的 handoff 副本
  → ConversationRuntime.ask → PlanningContext → 当前 Step 原始请求
  → General Worker 的 attachment_to_text / ocr_image
  → /artifacts 下的 Markdown、JSON 和有来源标注的文字预览
```

原件保存在 Git 忽略的 `.agent/feishu-attachments/objects/`，登记表为同目录
`attachments.sqlite3`。模型只获得 `/handoff/inbound/<attachment_id>/<filename>`
虚拟路径及文件名、格式、大小、附件编号，不获得飞书密钥或任意主机路径。
同名文件放在不同编号目录，不互相覆盖。附件副本位于当前 run 的 handoff，现有
General/Code 文件边界继续生效；Code Worker 没有新增宿主 OCR 工具，阅读/OCR
步骤安排给 General Worker。原始请求中的清单会继续传入 Step 指令，避免只靠
Scheduler 自己复述文件路径。

这是一段**本地 Python 接收适配代码**，复用已安装的 Feishu SDK 和现有 LangChain
原生工具，不新增 MCP Server、付费 API 或长期 OCR 服务。
接收时仅运行格式探测子进程，不加载 OCR 模型；Worker 真正读文档时才按
[本地 OCR 约定](local-ocr.md)加载已有中英 OCR。一份文档的一次调用内复用引擎，
调用结束或超时后释放进程。图片/PDF 能提取文字，不表示支持场景描述或视频理解。

## 重试、恢复与限制

- 根据发送者、聊天、消息 ID 生成稳定 Event ID。重复的普通任务、insert/replace
  消息复用原记录；并发重复推送也只创建一个任务。原有控制命令仍沿用原入口，
  不在本阶段承诺它们的跨重启去重。
- 消息下载过程中切换会话，已接收任务仍归属于开始接收时的 Conversation。
  插入任务不会因会话切换而去暂停另一个 Conversation 的任务。
- 附件关联先落盘，再创建队列事件；已入队任务的恢复无需重新向飞书下载。
  附件或副本校验失败、输入登记丢失时停止执行，不静默换文件。
- 两套 SQLite 登记并非一个分布式事务。下载或入队失败可能留下已下载、尚未进入
  队列的缓存，但不会产生一个先运行、后补附件的任务。未新增自动清理策略。
- 接收 PNG/JPEG 等可解码图片、PDF、PPTX、DOCX、XLSX，以及 UTF-8 TXT/MD/HTML/CSV。
  PDF/Office/图片按内容识别，改名为 `.bin` 也可识别；未知二进制、普通 ZIP、
  音视频不自动交给 OCR。旧版 DOC/PPT/XLS 不在此接收列表中。
- 每条消息最多 8 个附件、单个最多 20 MiB、合计最多 40 MiB；下载等待 60 秒。
  当前 SDK 会先缓冲下载结果，因此 20 MiB 是**下载完成后的接收限制**，不是
  网络流量或进程峰值内存的硬上限；超时取消也不保证 SDK 底层线程立即停止。
- 格式探测不是完整文档有效性验证。加密/损坏文档、过大的图片或复杂版式，仍可能
  在实际阅读时失败或返回警告。PDF 阅读默认每次最多 20 页，按 `next_page` 继续。
- 文件只落盘不执行。文件名和正文始终作为不可信内容传给模型。

## 验证与上线边界

本地测试使用安装中的 `lark-channel-sdk 1.4.0` 消息结构、真实 SQLite/EventRunPump、
真实 GeneralStepRuntime/LangChain 工具调用以及真实 CPU OCR。仅替换飞书网络下载、
对外消息发送和付费模型；测试入口函数从当前 `main.py` 提取，避免导入时启动真实服务。

```powershell
C:/ProgramData/anaconda3/envs/agent/python.exe -m pytest tests/test_feishu_attachments.py tests/test_event_models.py tests/test_event_store.py tests/test_event_run_pump.py tests/test_local_ocr.py tests/test_local_native_tools.py tests/test_general_worker.py tests/test_general_runtime.py tests/test_web_runtime.py tests/test_web_artifacts.py -q
```

覆盖单文件确认、富文本图片、回复/编号引用、并发去重、同名文件、会话隔离、
失败/大小限制、原消息 ID 下载、队列持久化后恢复、General 实际阅读、PDF 原生
文字＋图片中英文 OCR、引擎复用/进程释放、Step 指令保留路径、控制命令快速响应。

最终结果：**88 项通过，41.57 秒**（其中本阶段新增 16 项接收链路测试）。
仅有 SDK 依赖的两条弃用警告。新增模块/测试和 OCR 子进程文件通过 Ruff 检查与
格式检查，main.py 通过语法及 F/E9 静态检查；没有顺带重排主程序已有代码风格。

尚未完成在线飞书验收：没有重启现有程序、检查租户后台权限或发送真实测试消息，
也没有调用 DeepSeek 验证模型的自主规划/选工具表现。新代码在下次正常启动时加载。
届时应从用户飞书账号发一张图片和一份 PDF，回复原消息要求读取，核对文件名、
识别文字和最终回复。机器人必须有消息接收及对应消息资源读取权限，并能收到
该聊天的消息；SDK 支持附件并不等同于当前租户已配置好这些权限。
