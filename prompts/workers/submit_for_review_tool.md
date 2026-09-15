提交最终结论、各成功标准的声明、真实工具证据 ID 和产物候选，供独立审核。

<!-- include: workers/handoff_submission -->
每条声明对应支持它的证据；证据不足的标准保留为未解决项，不能当作已经通过。
若发现当前Step的对象关系、条件归属、集合运算或读写效果与用户原话实质冲突，填写plan_challenge并停止写入，等待独立Reviewer核验；接口不熟、单次调用失败或执行困难不是计划异议。

若提供HARNESS_CRITERIA，criterion_claims引用当前C编号，证据引用E编号；不要重写验收条件。交接须区分已确认事实、未尝试和真实失败，说明缺口与可读取的材料。

短例：HARNESS_CRITERIA 给出 C1、C2 时，submission.criterion_claims 必须各填一次；已完成项引用真实 E 编号，未完成项如实写缺口。不能省略整张验收表，也不能自造 C3 或字段名。
