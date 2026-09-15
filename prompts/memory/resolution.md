比较候选与旧记忆：ADD 新增，COEXIST 共存，MERGE 合并同一事实，SUPERSEDE 明确替代，IGNORE 不保存。先写 reason 说明新旧事实、时间与选择依据，再填写 action。MERGE/SUPERSEDE 的 target_memory_ids 只用输入中的 ID；其他动作留空。考虑有效时间，不创造事实。时间用带时区 ISO 8601，未知用 null。类型仅 profile、preference、relationship、project、task。图关系已经由宿主按固定枚举生成，你不能新增或改写关系。只输出 JSON：
示例：{"reason":"候选明确表达一项新增长期偏好","action":"ADD","target_memory_ids":[],"final_content":"用户偏好中文回答","final_memory_type":"preference","final_valid_from":null,"final_expires_at":null}
合并时保留仍有效的细节，不把不同时间的状态混为一个事实。新旧完全重复、没有增量信息时使用 IGNORE。
示例：旧记忆 ID 为 m1，内容“用户住杭州”；候选“用户现在搬到南京” → SUPERSEDE，target_memory_ids 为 ["m1"]，最终内容为“用户现居南京”。
若候选说“下个月可能搬到南京”，尚不能替代现居地；可用 COEXIST 单独保留带“可能”的计划，target_memory_ids 留空。例子中的 ID 仅用于说明，实际输出必须取自输入。

输出格式（JSON Schema）：
{{schema}}
