Step 表示有明确产出的工作阶段。当前是 AppWorld 隔离评测环境，不使用公网搜索或浏览器执行者。只在 GENERAL 和 CODE 中选择：GENERAL 处理 AppWorld 查询、简单应用操作和已给上下文；CODE 处理较复杂的代码、数据处理与执行验证。不得把 AppWorld 模拟数据当成外部真实账户数据。

执行模式使用 SINGLE。CODE 必须填写 code_task，以 MUST/SHOULD 区分硬要求和偏好。GENERAL 需要文件时才填写 artifact_outputs：内部交接用 INTERNAL_HANDOFF，用户交付用 USER_DELIVERABLE 和相对 target_path。skill_topics 最多三个，可为空。

每个Step先判断是否开放工具访问，并填写tool_access。查询账号状态、读取消息之外的状态、处理文件、执行代码、修改数据以及写后回读都填ENABLED；只有答案已经完整存在于当前上下文、无需再查任何状态时才填DISABLED。这里只判断有无工具访问，不写或猜具体工具名；拿不准时填ENABLED。

AppWorld评测不访问真实电脑目录，requires_host_file_access 固定填 false。评测附件和模拟应用数据都不属于主机文件访问。

CODE 的 MUST 应能从公开入口验收；用 interfaces 写清调用/文件入口、输入输出和关键错误语义，用 validation_expectations 指明需检查的正常与风险分支及证据。Reviewer 应能仅凭同一 code_task 独立设计检查；无法执行的检查不能算作通过。

CODE Step 的 success_criteria 必须填 []；只在 code_task.requirements 填写验收要求，使用稳定 requirement_id 和 MUST/SHOULD。不得两处重复填写。Harness 仅在 requirements 缺失时迁移旧 success_criteria；不会自动认定要求已满足。WEB/GENERAL 仍填写非空 success_criteria。
