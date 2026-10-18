# 用户可见对话的滚动 JSON 摘要

## 配置

`CONVERSATION_COMPACTION_ENABLED=false` 默认关闭；配置定义和开关校验在
`config.py`。关闭时输入保留全部完整对话原文，已有摘要不替代原文。
以下滚动摘要策略仅在显式开启后执行。

```dotenv
SUMMARY_LLM_PROVIDER=qwen
CONVERSATION_COMPACTION_ENABLED=false
SUMMARY_LLM_MODEL=qwen3.8-flash
PLANNING_HARD_RECENT_DIALOGUE_TURNS=3
PLANNING_CONVERSATION_SUMMARY_TRIGGER_TOKENS=10000
PLANNING_CONVERSATION_SUMMARY_MAX_CHARS=6000
```

修改 `.env` 后重新启动应用才生效。本次修改了本机对应配置，但未重启正式服务。
旧的 `PLANNING_HARD_RECENT_DIALOGUE_MAX_CHARS` 和
`PLANNING_CONVERSATION_SUMMARY_TRIGGER_TURNS` 保留解析兼容，不再控制此链路。

## 数据和顺序

完整历史仍在 Conversation 的 checkpoint 中，包含用户消息与最终发给用户的助手回复。
`ConversationRuntime._prepare_planning_context` 每次根据该历史组装 PlanningContextPack：

1. 固定保留最近配置轮数的用户/最终回复原文，不使用 Cross Encoder 挑选历史轮次，也不裁剪这些消息。
2. 较早、尚未摘要的轮次不足配置 token 门槛时，全部原文进入 `recent_dialogue`。
3. 达到门槛时，把旧 JSON 摘要及新增较早完整轮次一起发送给摘要模型。
4. Harness修复缺省/空容器等轻量错误，校验Schema、实质内容非空、输出未截断及字符上限。核心内容异常时最多修复重试一次。通过后写入摘要和覆盖轮数；原始messages不删除。
5. 下一次只处理覆盖轮数之后的新增较早轮次，同时带上旧摘要。
6. 两次仍失败时保留旧摘要和全部待摘要原文，不推进覆盖位置；下次进入对话压缩检查点再尝试。

门槛统计只针对新增待摘要轮次，不包括固定保留的近期轮次或旧摘要。
使用本地 tiktoken `cl100k_base` 编码统计文本，附加角色开销；这是确定性的 token **估算**，
不是 Qwen 专用 tokenizer 的精确计数或账单用量，也不是整个模型请求的容量保证。
摘要失败或单轮极长时不静默截断原文，因此仍可能接近模型上下文容量。

新链路不再重复注入所有历史用户原话，也不默认生成/注入 `previous_run_summary`。
用户原话仍完整保存在 checkpoint；已覆盖的较早用户要求进入 JSON 摘要。
内部 Step Report、Final Review、工具结果仍属于独立的 planning run，不进入此对话压缩输入。
没有新增历史查询工具或工具组。原有长期记忆及业务工具组路由不受此改动影响。

## JSON 协议

模型输出 `ConversationSummary`：

```json
{
  "user_requirements": ["仅导出 CSV，已取消 PDF"],
  "decisions": [],
  "assistant_results": ["助手报告已实现导出，测试尚未执行"],
  "pending_items": ["验证导出结果"],
  "important_references": ["export.py"]
}
```

所有字段必须存在且为字符串数组，禁止额外字段。输出复用
`structured_output_policy.with_provider_structured_output`：项目认定的 Qwen3.8-Flash
路径通过 API 参数发送 `response_format=json_schema` 与 `strict=true`，完整 Schema
只在接口中发送一次，不重复粘贴到文字提示。未验证的供应商使用该公共适配的既有
JSON-mode 与文字 Schema fallback。所有返回值仍经本地 Pydantic 校验，不按字符截断 JSON。
Harness 在保存时添加 `schema_version: 1` 和 `covered_turns`；覆盖边界不由模型决定。

摘要Schema统一在 `schemas/compaction.py`，公共校验/一次重试在 `compaction_runtime.py`；
token估算与对话摘要组装在 `conversation_compaction.py`；
自然语言摘要规则在 `prompts/conversation/summary.md`；
触发与 checkpoint 写入在 `conversation_runtime.py`。

旧的纯文字摘要/助手消息计数不会被认作新版覆盖边界。迁移时从原始消息重新开始，
若未达门槛就保留全部历史原文；已存在的旧摘要和记录不删除。
调整近期保留轮数变大时，相关近期原文会重新提供，可能与已有摘要少量重叠。

## 离线验证

`tests/test_conversation_compaction.py` 覆盖门槛、滚动更新、失败、旧 checkpoint、
配置调整、中英文 token 估算、16轮模拟、真实 SQLite 恢复及真实 SDK 在模拟 HTTP 传输上的严格 Schema 请求参数。
测试使用模拟摘要模型，没有云端调用；不能据此宣称摘要内容不会遗漏或误述。


## 单次任务内部压缩

见 `docs/internal-compaction.md`。Scheduler默认关闭长度清理；Worker保留已有默认关闭
开关，其启用路径已升级为严格JSON Schema。两种模型摘要共用一次修复重试策略。
