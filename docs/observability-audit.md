# Phoenix 审计与选型

审计日期：2026-08-31。项目：PersonalOps Agent。
数据来源：本地 .agent/phoenix/phoenix.db，含 WAL 的一致性只读快照。
范围：扫描全部 12 条 trace / 1,542 个 span，进一步检查全部 4 个 conversation_turn 的结果字段、云端调用身份和工具错误封装。
这是历史运行审计，不是 AppWorld 成绩，也没有重跑现有 Agent。未向任何云观测平台上传数据。

## 结论

本阶段保留 Phoenix，把时间用在独立验收、Token 口径、错误分类与实验关联上。没有证据表明当前问题必须靠换平台解决。

用户所说 LangChain 相关观测平台很可能是 LangSmith，但无法凭现有上下文确认其所读书籍。Phoenix 已经接入当前 LangChain/LangGraph；二者不是只能二选一的编排框架。

| 维度 | Phoenix | LangSmith | 当前决定 |
| --- | --- | --- | --- |
| 追踪链路与调用 | 当前项目已有真实记录 | 支持追踪和反馈 | 不为重新得到已有记录而迁移 |
| 评估与实验 | 支持代码评估器、模型评估器与实验 | 支持数据集、实验与反馈 | 官方 AppWorld 判分作为成绩真值，平台展示结果 |
| 图内调试 | 现有 trace 已带节点和执行信息 | Studio 支持节点提示词修改、trace 调试、加入数据集 | 若后续确实需要该工作流再验证收益 |
| 本地数据 | 当前数据库保留在本机 | 官方自托管属于 Enterprise 附加选项；托管版另有数据处理考虑 | 十天冲刺先保留本地轨迹 |

资料核对：[Phoenix evaluation](https://arize.com/docs/phoenix/evaluation/evals)、[LangSmith Studio](https://docs.langchain.com/langsmith/observability-studio)、[LangSmith self-hosted](https://docs.langchain.com/langsmith/self-hosted)。
上述是官方平台能力，不代表本地安装的每项 SDK 接口已验证。

## 实测结果

数据库记录时间为 2026-07-26 12:06:49 至 12:43:36（原始存储值，未在报告中推定其时区）。它不是最近一天的运行结果。
全部 12 条 trace 中只有 4 条是用户会话回合，其余是 4 条嵌入、3 条记忆整理和 1 条浏览器关闭。不能把 12 当作 12 道独立任务。

| 项目 | 实测值 | 含义 |
| --- | --- | --- |
| Span 状态 | OK 1,541 / ERROR 1 | 不能转换成任务成功率 |
| 云模型调用 span | ChatDeepSeek 128 + ChatCompletion 128 | 同一次调用被两层追踪记录 |
| 按 provider/model/response ID 关联 | 128 个唯一调用；128 个额外记录；0 个冲突或未识别项 | 本次可保守去重；不能只按相同 Token 数合并 |
| 去重后云端输入/输出 Token | 560,913 / 86,607 | 合计 647,520；仅 provider 返回的云端 usage |
| 直接累加全部云 LLM span | 1,295,040 Token | 恰好翻倍，不表示供应商真的多收一倍 |
| 本地 router | 40 次；0 次带标准 Token usage 属性 | 数据库列显示 0 不等于实测消耗为 0 |
| 费用记录 | 256 条，total_cost 非空为 0 条 | 不能宣称已准确统计费用 |
| 工具返回明确 error | 13 次，外层 span 全部 OK | 9 次 browser_navigate、3 次 browser_snapshot、1 次 shell |
| 标注/数据集/实验 | span_annotations、trace_annotations、datasets、experiments、experiment_runs 均 0 | 该库没有可用独立评估或实验成绩 |

费用需要模型和 usage 与价格表匹配。模型返回的缓存命中/未命中信息也应纳入口径；reasoning tokens 若已经包含在 completion 内不能重复相加。[Phoenix cost tracking](https://arize.com/docs/phoenix/tracing/how-to-tracing/cost-tracking)

## 四个会话回合的核查

以下 rowid 仅是本地审计定位号，不是 AppWorld 任务 ID。

| trace rowid | 耗时 | span 状态 | Agent 自报结果 | 停止原因 | 工具返回错误数 |
| --- | --- | --- | --- | --- | --- |
| 3 | 70.906 秒 | ERROR | 无 | InvalidUpdateError: Ambiguous update, specify as_node | 0 |
| 5 | 472.190 秒 | OK | PARTIAL | budget_exhausted_after_report；报告模型轮数 30、工具调用 38 | 3 |
| 8 | 384.617 秒 | OK | PARTIAL | step_failed_after_attempts；报告模型轮数 28、工具调用 41 | 9 |
| 11 | 214.527 秒 | OK | COMPLETED | all_steps_reported；报告模型轮数 23、工具调用 24 | 1 |

COMPLETED 仍是 Agent 自报，不能当成独立验收成功。回合中一次工具失败也不必然导致整题失败，可能恢复成功。

工具错误不是用全文搜索 error 得出的：读取了 ToolMessage 的 data.status。网页内容提到 error 并不构成工具失败。已检查的浏览器错误包括 ERR_ABORTED 和快照选择器解析异常。
root span OK 只说明调用正常返回；不能证明任务目标达成。这种分层本身不一定是追踪库 bug，缺的是另设业务结果和验收指标。

历史 checkpoint 异常不能直接断言当前代码仍未修复：现有 conversation_runtime.py 的状态更新已包含 as_node 参数。需隔离复现后才能决定是否改动。

当前只支持这批历史证据中的结论：长任务存在工具失败、预算耗尽/尝试失败，同时计费口径和独立评分缺失。尚不能判定是 Skill、schema、网络、模型还是重规划策略贡献最大；要在固定环境下逐项实验。

## 需要补齐的字段和检测

一题一个 trial_id，关联 benchmark_version、split、task_id、experiment_id、repeat_id、git_sha、model_config_hash、skill_hash、tool_schema_hash、budget_config。当前 metadata 主要是会话和 LangGraph 节点信息，没有上述完整实验标识。

评分分三层，互不替代：
1. 基础设施结果：超时、进程异常、数据库/执行环境不可用。
2. 工具/策略诊断：参数校验失败、工具明确失败、重复尝试、预算耗尽、错误恢复。
3. 独立任务评分：AppWorld 官方 evaluator 验收世界状态；不使用 Agent 的自我评价代替。

优先新增诊断：
- provider response ID 关联，云端 usage 仅计一次；本地模型缺失 usage 标为 unknown。
- 工具 status/error、重试次数和恢复结果单独统计，不能靠 span OK 汇总业务成功。
- 记录阶段耗时和各角色调用，分清执行器、规划/监督、摘要、记忆、优化器、评估器成本。
- 重复同工具同参数只能标为“待查无进展循环”；结合状态是否变化判断，不能把合理重试全部判错。
- 上下文压缩记录触发、前后 Token、保留约束和压缩后行为。此库没有独立 summary 更新事件，只有中间件调用，不能据此声称压缩有效或从未发生。
- 任务结束后的“没改坏其他数据”由官方验收检查，不靠文本好坏评分。
- 元数据关联不包含 API key。个人会话原文及 AppWorld 受限制题目内容不得为展示而公开上传。

## 复跑方式与限制

在仓库根目录执行：

```text
python scripts/audit_phoenix.py
python -m unittest discover -s tests -p test_phoenix_audit.py -v
```

本次使用 C:/Python313/python.exe 实际运行。脚本以 mode=ro 和 query_only 打开数据库，无网络、模型调用或 Phoenix 运行时依赖；仅输出计数与结果字段，不输出会话原文、session ID 或供应商请求 ID。
读取 JSONB 数据需要支持 JSONB 的 SQLite；未来 Phoenix schema 改动需适配，不能把私有 SQLite 表当永久公共 API。未来在线采集优先官方导出/API。

7 个测试通过：双层追踪去重、相同 usage 的不同调用不误合并、身份缺失/usage 冲突拒绝给总量、本地缺失计数、OK span 内工具失败、正文 error 不误报、自报完成不成为独立成功。
生产 Agent 的观测行为尚未修改；报告去重不代表 Phoenix UI 本身已经修复。

README 的默认端口仍写 6006，observability.py 与 phoenix_runtime.py 默认 6007；是文档不一致，不能据此判定现有环境的实际端口。

