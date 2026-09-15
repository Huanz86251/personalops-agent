提交当前 scheduler_epoch 的继续执行结果。REVISION_READY 提高版本；SUBMISSION_UPDATED 只改清单、版本不变；二者附完整 updated_submission。BLOCKED 或 REQUEST_SCOPE_CHANGE 说明原因，updated_submission 为 null。

提供 updated_submission 时遵循以下交接要求；其为 null 时仅如实报告阻碍。
<!-- include: workers/handoff_submission -->
