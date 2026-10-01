# Prompt 分层与角色内技能准备

> 本文描述 Skill 方法选择。工具组能力选择是另一条链路，已改为常驻 Cross-Encoder，见 [toolset-routing.md](toolset-routing.md)；两者不要混称。

最新逐句策略、MCP实测、Reporter字段及校验闭环见 [行动策略审查](skill-policy-action-audit.md)。没有增加运行中热切换；旧角色快照仍保存当时正文。

当前流程是先选择方法，再正式处理任务。Scheduler 的准备调用与正式规划调用分开；General、Web、Code、Code Reviewer 各自拥有准备节点和快照。General 保持轻量 LangChain、自检后直接报告；Web、Code 保持现有 Deep Agents 和审核流程。

2026-09-06 最新：38项方法统一重命名，不保留旧ID别名；Web把生活服务、场馆、影院、路线、网店报价分开，按真实Docker演练完善Code/Reviewer方法和路径协议。目录与选用边界见 [当前说明](skill-scenarios.md)，运行证据见 [角色实测](skill-role-field-test.md)。一次选择和固定快照机制不变；邮件禁止发送。

## Prompt 如何管理

- `prompts/planning/scheduler.md`：唯一 Scheduler 固定指令；同目录其余文件为按需追加的阶段协议，共享 Step 规则只披露一次。
- `prompts/workers/`：各 Worker 的职责与提交协议，保留角色之间的差异。
- `prompts/reviewers/code.md`：独立代码审核协议。
- `prompts/reporters/`：Step Reporter 与结构化输出协议。
- `prompts/runtime/`：运行时交接、选中技能的引导句。
- `prompts/tools/`：简短工具组说明及框架文件工具的中文描述；参数 Schema 保留英文。
- `skills/<领域>/<名称>/SKILL.md`：按需选择的领域方法。工具的准确参数、权限、提交和验收门槛不由技能定义。

Scheduler 的 system 只包含固定指令与选中技能。任务、计划、报告和阶段协议按顺序追加；Schema 类型、字段、枚举、约束不变，省去生成的标题和字段解释。相同协议仍在可见历史中时不再注入。Step Reporter 继续拆分固定协议和独立审核材料，不接入 Scheduler 的对话历史。已无调用者的 common 片段已移除。

## 准备、执行与恢复

1. 一个 Planning run 首次进入时，使用 Deep Agents `SkillsMiddleware` 解析标准 frontmatter，并冻结当前目录的元数据与正文。正文此时在应用状态中，未全部发送给模型。
2. Scheduler 准备调用只收到任务和相关上文、候选技能的 ID 与简介，返回零到三个技能 ID 和简短依据。固定角色规则随后连同选中正文、原始用户任务交给正式规划调用。
3. Scheduler 可在 Step 中指定最多三个 `skill_topics`。Worker 候选集合由 common、角色目录、这些主题取并集，再按角色与实际配置的工具过滤。主题不会开放新工具。
4. Worker 在其图的 before_agent 节点准备自己的技能；Code Reviewer 独立选择，不继承 Worker 的选择或隐藏推理。
5. 快照保存选中正文、角色、模式、目录哈希、逐技能内容哈希、整体快照哈希、选择方式与调用数。选择聊天记录不会拼入执行对话。准备调用仍进入正常模型回调和 `skills.prepare` trace。
6. 普通返修、CONTINUE 和 checkpoint 恢复沿用快照；不再次读取新版本替换旧内容。新 Planning run 才采用更新后的目录。恢复时发现快照被修改或技能依赖工具不可用会明确报错。

Scheduler 准备有单独的 LangGraph 节点，准备完就可以持久化，再进入正式规划。规划、Web 进度指导、Code 继续决策、重规划和最终验收共用一个 Scheduler 会话；最终验收不再独立选择技能。Step Reporter 从受限任务契约和最近三项有界执行摘要选择方法，正式审核仍使用 StepReviewPacket；未引用的 Worker 原始对话不向它开放。

Code 候选、返修状态和相关记忆的变化，以带 Harness 标记的消息追加到历史中。Code 按字段组追加变化，已有任务契约不重复发送；检查点 ID 等内部字段不发给模型。固定技能仍位于 system 前缀，正文不包含 YAML 元数据与内容哈希，完整快照保留在程序状态中。

## 会话与压缩

`PlanningContextPack.scheduler_session` 随图节点检查点保存；运行中的 Web 指导回调使用同一会话。节点执行期间进程崩溃，尚未随节点落入检查点的消息不保证恢复，本轮未增加逐消息持久化日志。每个新 Planning run 建立新 Scheduler 会话；跨任务对话仍使用现有上下文准备逻辑，没有机械截断成最近三条。

- `SCHEDULER_SUMMARY_TRIGGER_TOKENS=20000`：按近似 Token 数触发整理，仅删除旧过程消息、背景和非当前阶段协议；原始任务、用户要求、计划、结构化决定、StepReport 和交接回执不改写。协议被清理后，下次使用重新披露。阈值是软触发点，不能承诺受保护材料多大都能压到阈值内。
- `WORKER_COMPACTION_ENABLED=false`：当前默认关闭 Worker/Code Reviewer 的模型摘要压缩，原始工具调用、成功返回和错误返回均保留在当前 Worker 对话中。旧的 `WORKER_SUMMARY_*` 参数仅在以后显式开启该实验开关时生效。跨 Worker 时，最新报告的有效交接项不受旧字符预算限制；仅更早报告按完整条目裁剪并保留历史读取入口。

普通角色策略替换 Deep Agents 默认 SummarizationMiddleware，General 使用同一显式中间件；不与默认摘要叠加。

## 模式和预算

- `dynamic`：普通应用默认。有候选且预算足够时，至多一次结构化选择调用；无候选、明确的简单问候或预算不足时不调用。选择器可以返回空集合。
- `fixed`：调用方通过 `PlanningContextPack.skill_fixed_ids` 提供按角色分组的 ID，例如 `{"general": ["general-read-documents"]}`；不调用选择模型。
- `off`：禁用可选技能；固定角色、权限、工具与验收协议依旧有效。

普通应用可使用 `SKILL_ROUTING_MODE` 设置默认模式；每次运行可由 `PlanningContextPack.skill_mode` 覆盖。技能选择与正式执行使用该角色的模型配置。选择失败不重试，记录失败类型，回退到基础角色能力。

准备调用计入原有模型预算。Worker 至少保留后续执行空间，Scheduler 和 Step Reporter 保留正式结构化调用与修复预算。普通角色新一轮调用重置本轮计数，但保留技能快照与历史；checkpoint 原地恢复由 LangGraph 继续已有节点。

## 更新和评测

技能使用标准 `name` / `description`，项目扩展放在字符串型 `metadata.roles`、`metadata.topics` 和 `metadata.required-tools` 中。同名技能跨目录冲突会报错。当前技能没有额外脚本或资源；未来增加辅助文件时，应先扩展快照内容与资源访问机制，不能让旧运行引用会变化的外部文件。

AppWorld 的已有 `--skill` 文本注入接口保留，动态路由默认关闭，避免悄悄改变 baseline。新增配置示例：

```text
--skill-routing off
--skill-routing fixed --fixed-skill general:appworld-execute-api
--skill-routing dynamic
```

这些是现有评测命令的附加参数，不会自行启动评测。结果保存技能模式、目录和角色快照；源码指纹包含技能、路由与 Worker 代码。原始快照和 trace 仍只存私有评测目录。

正式对照应固定任务、模型、工具、预算与角色协议，再比较三种模式的官方完成结果、总调用、耗时、输入/缓存 token。此次 prompt/General 接线变动后的结果不能直接归因为相对旧 baseline 的技能收益。

## 当前边界

- 未运行收费模型或官方批量评测，尚无技能效果、延迟或缓存收益结论。
- 强制进度/收尾仍会收窄工具列表；提醒改成持久追加消息，system 不随进度切换。压缩也会改变消息前缀，未承诺全程缓存命中。
- 第一版在 attempt 内不做模型自主换技能；需要新方法时由新 attempt 或新任务采用新快照。
- Deep Agents 负责标准技能发现；项目的短准备调用与固定正文注入负责选择/执行隔离。没有开启一个可访问完整技能库、自由改写技能的额外 Agent。

主要代码入口：`scheduler_runtime.py`、`hard_planning.py`、`planning_graph.py`、`workers/compaction.py`、`skill_runtime/preparation.py`、`workers/code_state.py`。
