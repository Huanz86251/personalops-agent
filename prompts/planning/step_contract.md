Step 表示有明确产出的工作阶段。GENERAL 处理查询和简单应用操作；CODE 处理代码实现与执行验证。执行模式使用 SINGLE。CODE 必须填写 code_task，以 MUST/SHOULD 区分硬要求和偏好。General 需要文件时才填写 artifact_outputs：内部交接用 INTERNAL_HANDOFF，用户交付用 USER_DELIVERABLE 和相对 target_path。skill_topics 最多三个，可为空。

每个Step先判断是否开放工具访问，并填写tool_access。查询账号状态、读取外部消息或文件、运行代码、修改数据以及写后回读都填ENABLED；只有答案已经完整存在于当前上下文、无需再查任何状态时才填DISABLED。例：用户贴出一句话并只要求改写，可填DISABLED；用户要求“看看收件箱中谁回复了并替我处理”，即使最终只需一句回复，也必须填ENABLED。这里只判断有无工具访问，不写或猜具体工具名；拿不准时填ENABLED。

CODE 的 MUST 应能从公开入口验收；用 interfaces 写清调用/文件入口、输入输出和关键错误语义，用 validation_expectations 指明需检查的正常与风险分支及证据。Reviewer 应能仅凭同一 code_task 独立设计检查；无法执行的检查不能算作通过。
