提交当前候选、全部 requirement_id 的结果、拟交付 proposed_artifact_paths 和真实自测证据。

<!-- include: workers/handoff_submission -->
区分已满足、未满足和未验证的要求；证据应能对应具体检查，不能只写“已完成”。
若冻结code_task与用户原始要求存在会改变实现范围或外部效果的实质冲突，填写plan_challenge并停止扩大实现，交由Code Reviewer独立核验；未知API、测试失败或实现困难不是计划异议。
