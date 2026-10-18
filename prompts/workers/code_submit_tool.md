提交当前候选、全部 requirement_id 的结果、拟交付 proposed_artifact_paths 和真实自测证据。proposed_artifact_paths 优先填写相对于 `/workspace` 的路径（例如 `个人名片.html`）；若误填 `/workspace/个人名片.html`，Harness 会安全归一化，中文文件名无需改名。

<!-- include: workers/handoff_submission -->
区分已满足、未满足和未验证的要求；证据应能对应具体检查，不能只写“已完成”。
