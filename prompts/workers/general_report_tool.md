提交自行检查后的结果。填写未完成项、真实证据和需要交接的生成文件或已登记下载；文件由宿主依据产物回执、完整性和交付策略处理，不会因自报完成而自动共享。

<!-- include: workers/handoff_submission -->
不要填写顶层 status；Harness 会根据 criterion_claims、unresolved_items和预算收尾原因确定。文件生成、发送失败和已取得 message_id 的交付成功分别说明。
未完成项写清“未尝试/实际失败 → 当前阻塞或缺项 → 下一步”；依据工具记录，不推测错误原因，不复制密码或令牌。

正常完成且没有执行错误或缺口时，Harness直接将本次提交转成StepReport，不再调用模型复写。未完成、预算收尾或执行异常时，Harness按剩余额度启动独立Reviewer；文件本身不会单独触发Reviewer。下一位执行者或Reviewer不继承你的对话：handoff_knowledge写明真实接口用法、参数来源、已做与未做；文件注明路径、用途、产物槽位与核验方式。未调用只能说未尝试；接口不可用必须有真实错误依据。不要输出凭证。

若提供HARNESS_CRITERIA，criterion_claims按C编号说明已经完成、尚缺什么，并引用真实E编号。下一位执行者没有你的对话：handoff_knowledge保留后续必要的已查接口签名、参数来源、结果结构、文件定位及未确认限制；不交接密码或令牌。

短例：若当前有 C1、C2，两项都必须出现。已完成且有证据时可提交 `criterion_claims=[{criterion_id:C1, evidence_tool_call_ids:[E2], conclusion:实际结果}, {criterion_id:C2, evidence_tool_call_ids:[E4], conclusion:实际结果}]`；每项先填真实工具结果编号，再写 conclusion。某项尚未完成也要保留该编号，在 conclusion 写清缺口，并同时放入 unresolved_items。绝对不能提交空 criterion_claims 冒充完成。
