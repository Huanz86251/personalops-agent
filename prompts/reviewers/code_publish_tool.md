发布已验证通过的 Worker 拟交付路径。路径应相对于 /workspace；Harness 也会把 `/workspace/个人名片.html` 确定性归一化为 `个人名片.html`，中文文件名无需改名。不得传 `/review`、宿主绝对路径或 `..`。无文件的 EVALUATION 可传空列表。成功后依据回执提交审核报告。
