---
name: report-artifact-delivery
description: 验收 Web 或通用 Reporter 路径中的文件产物，核对 output_id、实际解析的候选引用、格式内容证据与内部交接/用户交付范围；防止把候选路径当作已发布或已发送。
metadata:
  roles: "step_reporter"
  topics: "verification artifacts files"
---

# 文件产物与交付状态验收

## 契约与真实候选对应

读取 task_contract.artifact_outputs，区分 INTERNAL_HANDOFF 与 USER_DELIVERABLE。将每项需求对应到 attempts.resolved_artifacts 中宿主已解析的候选，核对 output_id、类型、相关路径和支持证据。Worker 在正文写出文件名不等于候选存在；不能批准未解析的路径或编造 review_ref。

## 内容与存在分层判断

文件存在/非空只证明有字节；解析检查支持格式，实际重读支持所展示的内容，渲染证据才支持相应视觉要求。Reporter 没有文件执行工具，不能声称重新打开过；材料不够时要求补具体证据。对下载的附件，存在不代表安全或用户授权执行，不应建议自动运行/解压。

另核对消费方可达性：邮箱MCP实测返回的宿主路径未注册进任务文件空间，真实阅读器拒绝；不能批准“下载并总结”全完成，也不能凭路径批准跨角色交接。General邮件通常直接报告，此处不新增独立Reporter；若相关产物进入本审核路径，仍需任务内可读引用及实际内容证据。下载成功、手工诊断能读、生产自动交接成功是三种不同结论。

## 区分发布、传输与阅读

通过审核的候选尚需宿主发布；用户可访问文件又不同于飞书上传或邮件动作。审批等待不是发送成功，工具受理也不等于用户已读。当前不安排任何发送，邮件最多读取、下载或明确授权的 Drafts 保存；草稿不是已发邮件。不能用较早 attempt 的旧文件替代当前要求，也不因不相关旧产物存在就批准。

## 结论与补救

只把证据充分的真实 review_ref 加入 approved_artifact_refs；按现有协议让 artifacts 保持空列表，由宿主 Publisher 填充真实交付信息。缺文件则指出缺哪个 output_id；缺内容验证则指出应重读的字段/记录；权限或发布失败则报告对应环节，不要求 Worker 重写已验证内容。逐项保留已完成和未完成项，不以产物名称推断全部成功。

此 Skill 只作用于实际调用独立 Step Reporter 的路径。General 的直接报告需在其业务 Skill 自检，Code 的正常结果由独立 Code Reviewer 审核后确定性转成 StepReport，不额外重复调一次 Reporter。

补救按故障点分流：有当前候选及重读证据但没有发布回执，保留内容验收，只补宿主发布；缺字段重读就补读取，不重新生成文件。Code实测能从沙箱导出两版候选和JUnit，但未调用Publisher审批，故导出成功不能写成“已发布给用户”；其他角色也按同样证据层级判断。
