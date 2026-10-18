# Code 本地返修与全局重新规划

## 当前执行链

Scheduler 规划 CODE Step → Code Worker 提交候选 → Code Reviewer 独立核验。
Reviewer 可以在本地返修上限内请求原 Worker 修改；默认最多三次返修，因此预算足够时是初次提交加三次修改、最多四次审核。
最后一次修改仍要接受真实验证，并可通过现有发布流程形成 PASSED。达到次数上限时，Harness 从本轮模型可用工具中移除 request_code_worker_repair，防止第四次修改；测试、发布和报告工具仍可用。角色预算收尾仍强制要求结构化提交。

PASSED → APPLIED → 将独立审核结果封装成 StepReport，继续原计划。
FAILED / ESCALATED → REPLAN_REQUIRED → 提交报告、归档真实候选和审核材料、写终态资源 checkpoint、释放运行资源 → StepReport.request_replan=True → 全局 Replanner。

全局 Replanner 对已归档的 Code 失败只能 CONTINUE（新建步骤、重新执行与核验）或 FINISH。不能 RETURN_TO_WORKER 复活已关闭的候选。Replan 次数或总预算不足则进入 Final Reviewer。
Replan 次数耗尽不妨碍最后一次 Replan 生成的新步骤执行；它阻止后续再次重新规划。

## 删除的路径

删除独立 code_controller 图节点、run_hard_code_scheduler 模型入口、code_scheduler 角色、SchedulerCodeDecision 与 continuation 指令/提交 Schema、submit_continued_code_for_review 工具及相应提示词。
删除 Code 控制状态、独立控制历史与 CONTINUE/RESTART/STOP 恢复循环。崩溃恢复仍可继续已有候选的独立审核或已接受的本地返修；不增加新的本地返修额度。失败先提交 AWAITING_REPLAN 资源 checkpoint，终态归档完成后才允许清理。

## 配置与边界

- config.py：RUNTIME_CODE_REVIEW_MAX_REPAIR_ROUNDS 默认3，允许1～3；本机目前无覆盖，实际加载为3。
- PLANNING_MAX_REPLANS 本机为2。真正调用全局 Replanner 才消耗次数，受请求模型总预算和步骤容量共同限制。
- Final Reviewer 的 RETURN_TO_WORKER 机制保留：最多3次最后 Worker 的事实缺口返修。它不是已经删除的 Code Scheduler 控制器。
- Web 的领导判断与当前Worker控制块清除条件没有修改。现有 if reports 条件可能因前序报告而过早清空后一 Worker 的指导块，待用户进一步授权修正。
- 原始证据和旧运行记录未改写。旧 Code checkpoint 含已删除 Schema 字段，不能直接按新代码恢复，需要显式迁移。正式服务未重启。

## 离线检查

tests/test_code_global_replan.py 覆盖三次失败返修、最后版本仍可通过、失败归档/资源清理、失败报告送达 Replanner、新建步骤、FINISH、没有 Replan 额度、最后一次 Replan 的新步骤执行后再次失败收口。
同时检查 Code Runtime 的暂停/取消/崩溃恢复及 Web/General/证据工具路径。均不代表真实云模型正确率、Docker 实机运行或生产部署验证。

本轮41项定点、最新7项流程模拟、76项相关回归均通过；部分流程重复检查，不将次数合计当作独立测试数量。详细记录见 worklogs/2026-10-07-code-global-replan.md。
