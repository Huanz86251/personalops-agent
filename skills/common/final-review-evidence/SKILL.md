---
name: final-review-evidence
description: 从用户原始要求、范围合同、计划、StepReport和最后Worker回执做最终证据审核；用于决定结案、交回原Worker或升级重规划。
metadata:
  roles: "final_reviewer"
  topics: "verification final-review"
---

# 最终证据审核

先按用户原话固定对象、范围、条件和例外，再逐条核对成功标准。计划和Worker结论只是待核对材料，不能替代真实证据。

每条标准先列证据编号和实际观测，再判断 MET、PARTIAL、NOT_MET 或 UNKNOWN。只要仍缺事实且原Worker保留执行上下文，就优先 RETURN_TO_WORKER；返修单只写“哪里没有通过、实际看到了什么、还缺什么”，不猜工具名、接口名、参数或操作步骤，让原Worker根据自己的真实工具上下文决定怎么补。

例：员工说“合同已经归档”，但记录只有上传回执，没有归档后的目录回读。审核员应写“上传已确认；没有观察到归档目录中的文件，缺少归档后回读”，交回同一员工。不能写“调用 move_contract(file_id=...)”，因为审核员没有执行上下文，不应替员工猜操作。

返修轮次用尽后，只有确实需要改变任务拆分、工具能力或执行路线时才 REPLAN。已经有足够证据时直接 FINAL，不为走流程制造返修。
