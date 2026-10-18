# Knowledge 交接与轻量引用检查

## 文件入口

- Schema：`schemas/handoff.py`，`HandoffKnowledge`。
- 公共提示词：`prompts/workers/handoff_submission.md`，报告工具通过 include 使用。
- Harness：`handoff_knowledge.py`，`check_handoff_knowledge`、`collect_handoff_knowledge`、`compact_direct_handoff`。
- 提交入口：General `workers/general_completion.py`；Web `workers/submission.py`；Code `workers/code_submission.py`。

## 模型填写什么

```json
{
  "topic": "读取价格文件",
  "source": "共享文件 prices.csv",
  "evidence_refs": ["E1", "E99"],
  "usage": "读取时使用 sep=\";\"。",
  "observed_result": "真实读取返回的表头是 brand;price。",
  "next_action": "读取 price 列并计算总价。"
}
```

source 是可重新读取的位置，evidence_refs 是当前 Worker 的真实依据。E 是普通工具结果，W 是特定网页结果；S1-1 是历史读取工具的短引用，二者用途不同，不自动互换。API/Python 函数也使用同一结构，在 usage 中说明名称/签名、参数、调用例子、限制和必要前提，observed_result 区分只查文档与实际执行。没有可复用知识时填[]；纯建议可不引用，不能把猜测写成已观察事实。

## Harness 怎么处理

1. 在 Worker 接受提交时，读取 state 中真实当前/归档消息及证据编号映射。
2. 独立解析每个引用，复用 resolve_tool_evidence：需要实际请求、非空结果，拒绝控制工具；不检查 API 名称，不执行调用，也不调用额外模型。
3. 有效引用恢复为底层调用 ID，保存调用名、参数、原始结果、结果状态与来源 Worker/步骤。短 E 编号不会被当作跨 Worker 全局 ID。
4. 未解析引用记录编号和错误，保留原交接正文。全部可解析为 RESOLVED；部分可解析为 PARTIAL；没有可解析依据为 UNSUPPORTED。失败工具返回仍可解析，仅支持失败事实，不能支持成功。
5. 检查字段由 Harness 生成，在模型参数 Schema 中隐藏；即便模型显式传入这些字段，提交时也会重新计算覆盖，不能自我认证。
6. 保存到 Worker submission / StepReport 后，collect_handoff_knowledge 收集被接受的 Worker 内容，不让 Reporter 复写知识正文。同 topic/source 的后续 Worker 版本覆盖旧版本。
7. 下一位默认任务文字中的 worker_notes 带正文、引用检查状态、来源元数据及未解析原因，不自动塞入完整工具结果。最新交接完整保留，较早交接受既有预算限制。原始结果保存在接受记录中；需要查看时使用当前任务历史工具按步骤选择调用再读原文。

General、Web、Code 提交都接入检查；Code Reviewer 提交知识也会检查，但 Worker 知识收集仍不采用 Reviewer 替写来覆盖 Worker 内容。Web 单独的报告封装及并行报告合并保留 knowledge。

## 证据边界

RESOLVED 证明引用找到真实调用/结果，不证明正文正确、参数有效、来源可信或任务完成。例：真实返回 permission denied，而模型写“读取成功”，引用仍能解析；矛盾需后续执行/审核处理。未引用的说明会保留并标 UNSUPPORTED，不触发整份报告重试。报告验收声明的原有证据约束仍独立执行，不因 knowledge 的宽容规则放松。

## 移除项与恢复注意

API 专用 handoff_apis、knowledge.apis、handoff_api_receipts、validated_apis/rejected_api_leads 和 api_handoff.py 已移除，未改成名称正则检查。旧测试由统一知识交接与引用测试替代。历史私有执行档案未改写；含旧 API 字段的 checkpoint 需要显式迁移，不能假设可按新 Schema 直接恢复。正式服务未重启。
