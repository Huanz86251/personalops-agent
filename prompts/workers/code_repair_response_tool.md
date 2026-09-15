回应返修。REVISION_READY 提高版本并附完整 updated_submission；只补清单用 SUBMISSION_UPDATED，版本不变。TEST_DISPUTE、BLOCKED、REQUEST_SCOPE_CHANGE 说明原因，updated_submission 为 null。

提供 updated_submission 时遵循以下交接要求；其为 null 时仅如实报告阻碍。
<!-- include: workers/handoff_submission -->
