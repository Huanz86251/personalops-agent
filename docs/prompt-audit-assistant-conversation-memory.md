# Assistant、Conversation、Memory 提示词审查

2026-09-05。根据当前源码逐份阅读并检查实际调用链。用户授权直接修改；本轮覆盖明确点名的三个目录，共 9 份原有提示词，以及承接 Assistant 规则的 common 片段。未运行付费模型、AppWorld 或优化批次。

## 逐文件结论

| 原文件 | 实际调用与时机 | 处理与原因 |
| --- | --- | --- |
| assistant/base.md | 仅被 build_runtime_prompt 引用，后者仅被无人调用的 build_context_middlewares 引用 | 删除这份旧入口及两个无调用函数。通用证据、权限范围规则归 common；中文回复规则接到真正产生用户回复的规划角色。保留 context_middlewares 中仍被调用的对话摘要与其他历史修复辅助代码。 |
| conversation/summary.md | ConversationRuntime._prepare_planning_context 在较早轮次达到阈值时调用 summarize_conversation_history | 继续留在 conversation；职责改为历史进展，取消重建当前目标和自行制定下一步的要求。输入没有独立工具轨迹，必须区分助手自报和验证事实。规则作为 system，历史作为独立 user 材料消息。 |
| conversation/title.md | agent.generate_conversation_title，由 ConversationRuntime 生成会话标题时调用 | 继续独立；标题是展示元数据，不进入任务约束。长度说明对齐代码的 10 字符限制，英文数字也计数。输入不得作为命令执行。 |
| memory/write_gate.md | classify_memory_write_level；确定性规则未命中时调用本地分类器，失败仍进入提取 | 保留初筛阶段，统一单标签说明，区分一次性要求与长期事实。移除代码中强制将“忘记/删除”送入写入的规则，避免伪装为遗忘入口。 |
| memory/frame_detection.md + memory/typed_extraction.md | 通过门控的候选先批量分帧，再在同一消息前缀后追加命中的typed Schema | 第一轮只输出candidate/frame ID与类型；第二轮输出规范化摘要和详细字段。原文证据由宿主回填，自由relation与旧单轮extraction已移除。 |
| memory/resolution_gate.md | 历史提示词；当前typed写入和启动预热均不再调用 | 冲突已改为`conflict_key + confidence + dedupe_key + validity overlap`宿主硬规则，文件仅保留历史审计。 |
| memory/resolution.md | 历史提示词；当前typed写入和`/clean`均不再调用 | 模型不再判冲突或决定胜者；用户A/B选择后宿主soft-retire并刷新冲突组。 |
| memory/read_gate.md | 历史兼容提示词；当前retrieve_for_turn使用Dense、BM25、图扩展和Cross-Encoder阈值完成读取门控，不再逐条调用生成式Router | 保留文件仅供兼容审计；当前读取链路不调用它。相关不代表当前真实或获得操作授权。 |
| memory/context.md | MemoryService.format_context 将筛选后的记忆和时间渲染为上下文 | 留在 memory，属于背景包装，非角色系统身份。明确不得替代当前原文、验收标准或权限，旧记忆可能过期。 |

这 9 份原有提示词的自然语言本来就是中文，本轮主要修正职责与流程。common/evidence 和 common/task_scope 原来为英文，已翻译并承接旧 base 的合理规则。所有原有 JSON 字段、枚举、分类标签和工具名保持英文。

## 用户要求与压缩的边界

旧实现已有 PlanningContextPack.user_request，一次规划及重规划中保留当前用户请求。因此不能笼统说旧系统把当前目标完全交给摘要保存。

真正缺口在跨轮历史：较早用户原文被归入滚动摘要，最近对话也按字符预算裁剪。原消息仍在会话状态中，但规划模型得到的视图可能只有被压缩的版本。摘要若漏掉约束，“存储里有原文”并不等于“模型能看到原文”。

现新增 user_instruction_history，从会话状态的用户消息按顺序构建，独立于摘要与最近对话裁剪；进入 PlanningContextPack 并随规划 checkpoint 序列化。Scheduler 技能准备与正式规划都得到它。当前 user_request 仍独立保留；Worker 继续通过已确定的任务合同执行，没有把全部对话直接灌给每个 Worker。

common/context_authority 明确当前修改和取消优先，历史保留不意味着旧任务恢复。该固定规则接入 supervisor、replanner、final_reviewer、code_controller。common/user_response 接入前三个会生成用户答复的角色。通用证据和任务范围沿用现有 include 引用，不建立新的总控 base。

## 仍然存在的边界

- 完整历史用户文本会增加长对话的输入长度，这是保留精确要求的明确代价。这里没有实现自动判定任务归属的指令账本，也没有声称无限上下文；后续若需要限定活动任务范围，应通过显式任务边界和原文引用设计处理，不能偷偷截断此字段。
- 新字段能从仍存在的会话原文建立。旧 checkpoint 的字段默认空以保证兼容；已经丢失的原文不能由摘要无损恢复。文本提取不等于保存附件二进制内容。
- 摘要及回退仍有长度限制，只保证历史材料压缩为有限长度，不能保证无损保留所有进展。
- Memory 当前没有用户遗忘/物理删除动作。这里只修正提示词与错误路由暗示，不声称新增了遗忘能力；模型也不能保证只靠文字规则绝不误提取。
- 没有测量服务商缓存命中率。摘要规则与材料分离让固定部分更稳定，但缓存是否命中仍由具体请求和服务商决定。

## 验证

新增离线测试覆盖长原文在摘要与近期裁剪后完整保留、摘要恢复不重复调用、上下文序列化、旧字段兼容、摘要规则与材料隔离、摘要失败回退和公共规则接入实际角色。使用项目 agent Python 环境；系统 Python313 仅作静态检查，不具备 LangChain 依赖。

结果：新增上下文与 Memory 协议测试 7 项通过，已有技能准备回归 10 项通过；7 个 Python 文件语法检查通过，49 份组合提示词加载及变量渲染成功。日志位于本地 `.agent/prompt-audit-context-tests.log` 与 `.agent/prompt-audit-skill-tests.log`。测试出现既有依赖弃用及 nul 文件 ResourceWarning；首次测试尝试向未启动的本地 Phoenix 导出 trace 失败，后续关闭测试追踪导出，测试未调用外部模型。
