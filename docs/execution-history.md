# 当前任务执行历史：按步骤查看调用，再读取结果

`read_execution_history` 的输入 Schema 位于 `schemas/execution_history.py`，工具说明位于 `prompts/workers/read_execution_history_tool.md`，实现位于 `workers/history_archive.py`，持久短引用索引位于 `workers/history_index.py`。

## 第一阶段：查看调用

```json
{"step_ids": [1, 2, 3]}
```

返回所选步骤的业务工具名称、参数、Worker 标识及 `S1-1` 等引用，不返回工具结果，也不返回助手思考文字。第一个数字是步骤编号，第二个数字是该步骤归档调用的顺序编号。重试、并行分支及 Reviewer 的调用归入对应步骤，Worker 标识仍可区分它们。步骤编号不是 thread_id；一个步骤可能涉及多个 thread。

## 第二阶段：读取原始结果

```json
{"references": ["S1-2", "S2-1"]}
```

仅返回命中调用的归档工具结果，不截断正文。有效 references 决定读取内容，不受 step_ids 限制。若调用尚未产生结果，返回 `PENDING_RESULT`。部分引用无效时只返回命中结果及 invalid_references，不返回整体清单。全部未命中或引用留空时返回 step_ids 所选调用清单；step_ids 省略/null/[] 时列出当前任务全部步骤。第一次查询指引明确要求 references 留空，step_ids 可填写多个步骤。不存在的步骤通过 available_step_ids 与空清单区分。

短引用通过任务目录内 SQLite 索引持久化，不由模型生成。底层 H 文件仍保留哈希完整性检查。索引不是证据来源，结果必须来自真实存在的归档请求和工具结果。跨任务读取不开放。该工具沿用现有 Worker / Code Reviewer 注册路径；Final Reviewer 本次未新增此工具。

工具参数本身或所选结果仍可能很长；本方案减少不需要的结果正文，不承诺固定返回 token 上限。旧的 mode/reference/分页参数不再是公开输入。

## 验证

2026-10-07 离线联合回归 70 项通过。覆盖选择性读取、无效引用回退、60k 字结果不截断、重试和并行同名调用、引用持久化、并发索引、任务隔离、待返回结果与归档篡改。另运行合成价格查询模拟，验证三个实际工具输入/输出路径。未调用云模型或真实网页。
