# 单次任务内部压缩

## 统一配置入口（2026-10-05更新）

配置定义和校验统一在 `config.py`，环境变量在 `.env` 设置、`.env.example` 查阅。
Scheduler 和跨轮对话分别使用 PlanningSettings 的独立开关；Worker 使用
WorkerCompactionSettings / load_worker_compaction_settings，门槛和保留数量也集中读取。
三个开关默认及本机均为 false：SCHEDULER_COMPACTION_ENABLED、
WORKER_COMPACTION_ENABLED、CONVERSATION_COMPACTION_ENABLED。
跨轮摘要关闭时不调用摘要模型、不更新摘要边界，并从canonical messages恢复全部
完整对话原文，已有摘要保留在存储中但不用于替换本次输入。

## Scheduler

`SCHEDULER_COMPACTION_ENABLED=false` 为默认值。本机 `.env` 已明确设为false。
配置读取到 `PlanningSettings.scheduler_compaction_enabled`，主图所有节点包装器均传递此开关。
关闭时，`SchedulerConversation.compact()` 不根据token门槛删任何记录。
当前阶段的协议提示仍按请求生命周期替换，这是协议切换，不是长度触发的历史清理。

如显式开启，`SCHEDULER_SUMMARY_TRIGGER_TOKENS` 控制原有确定性过程清理。
全部未摘要对话（含配置保留的最近轮次）和背景摘要均为受保护记录。
旧checkpoint的较早对话/辅助背景保护标记会升级；任务、已审核步骤、StepReport、
产物交付等事实继续保护。修复无活动协议时 `None == None` 导致无key过程记录
被误判为活动协议的旧问题。

这是优先保留决策依据的设计取舍；没有做清理开启/关闭的任务准确率对照试验。

## Worker

### 动态预算与摘要输入边界（2026-10-05）

四类生产Worker/Reviewer的压缩器放在项目middleware列表末端，使用wrap_model_call，
在任务运行时指示、工具过滤/收尾选择完成后检查ModelRequest。DeepAgents仍可添加
自身中间件；本预算是项目末端请求的近似估算，并非供应商精确账单token。
tools与response_format是请求独立参数，不是messages最上/最下的一段文字。
预算计入系统消息、本轮工具/输出Schema、tool_choice及消息历史；触发门槛也看此完整估算。

新增config.py配置（默认）：
WORKER_COMPACTION_TARGET_TOKENS=20000；WORKER_COMPACTION_SUMMARY_RATIO=0.16；
WORKER_COMPACTION_PROTECTED_RATIO_LIMIT=0.70；WORKER_COMPACTION_SAFETY_TOKENS=512。
不可压缩占比按“不可压缩估算/本轮总估算”计算，达到70%跳过本轮。
摘要目标=min(目标总量×比例, 目标总量−不可压缩量−安全余量)，无正额度则跳过。

用户/Scheduler任务、运行时指示、系统提示、本轮工具定义不进入摘要输入；较早执行
交互、旧摘要和带knowledge_source标记的较早RAG消息可进入，最近完整工具交互保留。
摘要只收到所选执行材料、相关证据ID和动态目标提示。保持JSON Schema输出；不硬截
字符串，不设置动态API生成上限，并移除专用摘要模型克隆继承的max_tokens/
max_completion_tokens，原执行模型不变。供应商自身硬上限仍存在。

本地检查摘要token和重组总量；超预算携带actual/target重试一次，仍失败保持原文。
可选引用仅保留模型选择的真实相关ID，未知/非法引用轻量过滤，可为空；原始RAG
与工具消息继续归档，不强制向摘要补回所有ID。skip不会展开旧摘要。
本策略不是硬窗口溢出恢复器：原请求不适合供应商窗口时仍可能返回超限错误。
飞书conversation策略没有本次变更，全部开关仍默认关闭。

保留默认关闭的 `WORKER_COMPACTION_ENABLED=false`。开启后的相关配置：

```dotenv
WORKER_SUMMARY_TRIGGER_TOKENS=10000
WORKER_SUMMARY_KEEP_MESSAGES=2
WORKER_SUMMARY_KEEP_TOKENS=4000
WORKER_SUMMARY_MAX_CHARS=6000
```

生产的General/Web/Code/Code Reviewer摘要角色当前均为Qwen3.8-Flash，仍支持各自角色配置。
达到估算门槛时，只选择可压缩的较早执行消息；保留用户/运行时要求、近期原文、
完整控制交接。普通工具调用及返回必须配对完整。旧生成的JSON执行摘要可再次合并，
不会永久累积多份摘要。原始工具消息保存在archive与既有执行档案中。

输出 `WorkerExecutionSummary` 含观察结果、错误失败、未完成工作、文件产物、
证据引用与下一步。Schema在 `schemas/compaction.py`，提示词在
`prompts/conversation/worker_summary.md`。公共严格接口在 `compaction_runtime.py`。
输入是Harness组装的JSON执行材料；输出Schema通过API参数传入，严格供应商提示词
不重复完整Schema。原始reasoning/response metadata不进入摘要材料。

## 轻量修复和一次重试

两种模型摘要采用同一规则：

- 缺省、null、空字符串字段补空数组，字符串标量可无损包装成一项。
- 可选元数据的非法容器补空值；核心字段非空但类型非法不可静默丢弃。
- 对话要求/决定/结果/待办全部为空，或Worker观察结果/失败/未完成全部为空，视为异常。
- 无效JSON、核心异常、输出截断、超长等最多修复重试一次。
- Worker引用ID由Harness依据选中材料校验，不接受模型发明的ID；允许省略无关ID。
- Worker还校验摘要确实缩短；不缩短也只重试一次。
- 两次失败后保留原文，Worker不替换messages、不新归档、不推进对话覆盖轮数。
- 同一份未变化的Worker输入失败后不立刻重复付费尝试；新消息出现后再检查。
- Worker预算保留两次摘要尝试与两次正常执行/收尾空间，两次尝试均计入预算。

API Schema规定所有键出现，语义上无内容的键可为空；Harness的缺省修复兼容异常返回，
不意味着正常云端请求省略required字段。结构校验不能证明摘要内容完全忠实。

## 验证边界

离线测试覆盖开关、保护规则、核心/可选错误、一次重试、失败回退、调用预算、
旧Worker摘要合并及真实SDK的模拟HTTP严格Schema负载。没有调用云模型验证摘要质量，
没有AppWorld或准确率A/B实验。配置在应用重启后生效，本次未重启正式服务。
