# 执行中提出计划异议

## 配置与工具

`config.WorkerRuntimeSettings.plan_objection_enabled` 对应环境变量
`RUNTIME_WORKER_PLAN_OBJECTION_ENABLED`，默认 `true`。
General、Web、Code Worker 在创建时按开关注册 `raise_plan_objection`；
关闭时不注册。这个开关独立于 Web 的 `require_progress_checkpoint`。
Reviewer 不注册此工具。

工具参数只有三项，都是非空字符串：

```json
{
  "problem": "用户要求读取CSV，当前步骤却要求访问数据库",
  "reason": "执行当前合同会改变数据来源，偏离用户原话",
  "suggested_direction": "重新规划为读取用户指定CSV并生成统计结果"
}
```

工具说明保存在 `prompts/workers/raise_plan_objection_tool.md`；
本地验证协议在 `schemas/worker_control.py`。
LangChain 从函数签名生成工具参数 Schema、注入 `ToolRuntime`，
工具内部再用 `WorkerPlanObjection` 校验三个字段。
没有把隐藏的运行时参数放进模型要填写的三个字段中。

仅用于规划与用户要求的实质冲突，或证据说明规划前提不成立。
API不熟、一次错误、普通实现困难仍属于当前 Worker 的执行工作。

## 一次调用发生什么

1. `agent.ask_worker` 在调用作用域中绑定本轮 Scheduler 回调。
   Code Runtime 的嵌套 Worker 调用继承该作用域。
2. Worker 调用工具。工具调用 Scheduler 的 `worker_objection` 协议，
   复用严格结构化输出、校验重试、Scheduler 会话锁和计费统计。
3. Scheduler 的提示词在 `prompts/planning/worker_objection.md`。
   它只做短决策，不在这次调用中填写完整 `PlanStep`。

| 决策 | 当前 Worker | 后续路径 | Replan 次数 |
| --- | --- | --- | --- |
| CONTINUE | 原上下文继续 | 工具回复说明异议未被接受 | 不消耗 |
| GUIDE | 原上下文继续 | 工具回复包含 `guidance` | 不消耗 |
| REPLAN | 停止 | 主图进入既有全局 Replanner | 按既有预算消耗 |
| FINISH | 停止 | 主图进入 Final Reviewer 汇报 | 不消耗 |

REPLAN 可用性由 Harness 提供。若模型在没有额度时仍选 REPLAN，
Harness 改为 GUIDE，并要求如实报告无法完成的部分。
Scheduler 模型预算不足或输出最终不能校验时保守返回 CONTINUE，
不会借这次工具调用绕过预算修改计划。模型调用仍计入执行额度。

CONTINUE/GUIDE 的原始请求及判定保存到 Worker State 的
`worker_objection_records`，并作为 ToolMessage 留在 Worker 对话中。
Scheduler 的本次事件/决策块属于临时协议块，结束后移除；
停止时主图产生控制报告，随后作为步骤事实进入 Scheduler 的正常投影。

## 停止与证据边界

REPLAN/FINISH 写入 `worker_objection_outcome`。
`WorkerObjectionMiddleware.before_model` 返回 `jump_to="end"`，
不再要求 Worker 为提出异议额外填写终末报告。
同一轮中的业务工具被拦截；同一轮多个异议仅执行第一个。
工具在强制进度、收尾的工具过滤中也保留；已结束或预算用尽的图不会继续调用模型。

主图用 `assessment_source="SCHEDULER_CONTROL"`、未验证标准 `UNKNOWN`
记录停止原因、Worker 自述问题和建议方向。
这不是独立 Reviewer 的通过结论，不会把异议自述当成已证实事实。
General Step Reporter / Code Reviewer 不作为这个控制动作的前置条件。
普通完成、失败、返修和终末降级审批仍走原有路径。

Code 在退出前导出候选和审核目录、提交终态记录，然后清理沙盒。
REPLAN 记为 SUPERSEDED，FINISH 记为 CANCELLED；没有审核通过和发布回执。
FINISH 后不能由 Final Reviewer 重新唤起这个已停止的 Worker。

## 旧协议

删除 `PlanChallenge`、Worker 提交中的 `plan_challenge`、Reviewer 的
`confirmed_plan_challenge` 以及旧 Replanner 的“驳回后恢复原 Worker”专用分支。
全局 Replanner 只生成后续步骤或 FINISH；它不负责恢复已经关闭的候选。
Final Reviewer 对普通任务的 RETURN_TO_WORKER 仍保留。
历史归档不重写；含删除字段的旧严格 Code checkpoint 恢复前需要显式迁移。

## Code 导入文件

`host_import_file`：路径策略检查 → 宿主私有副本 → 大小/SHA256检查 →
上传沙盒成功 → 返回 IMPORTED 和 `execution_path`。
它仍是只读导入，不触发外发审批，也不授权修改或发送宿主原文件。

```json
{
  "status": "IMPORTED",
  "reading_path": "/downloads/file-example",
  "execution_path": "/workspace/.private-downloads/file-example/run_me.py"
}
```

此时可直接 `execute("python /workspace/.private-downloads/file-example/run_me.py")`。
不需要先 `read_file`。读引用仍核对任务、Worker及原登记副本完整性，
只解析已导入的沙盒文件，不在读取时上传。
Reviewer 导入根为 `/review/.private-downloads`，避免写只读候选区。
无沙盒的 General/Web 只保存自己的宿主私有副本，`execution_path=null`。
策略拒绝返回 POLICY_DENIED，复制失败返回 IMPORT_FAILED；失败不登记为导入成功。
旧引用若没有沙盒路径，Code 读取会要求重新导入，而不会偷偷复制。

导入目标通过 `task_file_imports.import_destination` 的 ContextVar 在工具执行时传递，
不会放进提示词或让模型自己选择 Worker/Reviewer 的存储根。
