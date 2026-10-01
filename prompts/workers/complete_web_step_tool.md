一次性完成当前 Web Step：提交逐项结论、真实证据、可交付正文、产物候选、错误、未解决项以及必要的重规划或降级建议。Harness 会在同一次工具调用中验证证据与产物并生成 StepReport；不会再启动 Web Reviewer，也不会要求第二次报告调用。

<!-- include: workers/handoff_submission -->
每条 criterion_claim 必须引用 HARNESS_CRITERIA 中的 C 编号，并填写 status、conclusion 和支持它的真实证据。普通工具证据引用 E 编号；Web 搜索优先引用 WA1/WB2 一类具体 source_ref，不能用查询页编号或搜索结果整包替代具体来源。MET 没有真实证据时会被 Harness 保守降为 UNKNOWN。

证据已经足够时立即调用本工具，不为润色继续搜索。漏填的冻结标准会由 Harness 补成 UNKNOWN；字符串误填列表会被包装成列表；可选说明字段格式错误时会被省略；超长 final_conclusion 会在不丢内容的前提下迁移到 answer_text。未知或重复的 C 编号会改变语义，仍需只修报告后重提，不得重新调用业务工具。

summary、confirmed_results、completed_work、errors、unresolved_items 和 next_action 应足够让 Final Reviewer 理解真实结果，但不要复制整段网页。需要改变原始要求时，填写 request_replan、replan_reason 和 recovery_proposal；Harness 会沿用既有用户确认链路，未获批准的降级不能冒充完成。

若任务要求文字交付，把完整的分点建议正文放入 answer_text；不要把整篇建议塞进 final_conclusion。只交付文件或当前 Step 不产生用户可见正文时，填写 answer_text=null。

例：用户要求“莱西今晚最火的爱情片”，现有证据只能支持“今晚可看且有公开评分的爱情片之一”，则“最火”对应项应为 PARTIAL 或 UNKNOWN，并可提出透明降级；不得直接写成 MET。
