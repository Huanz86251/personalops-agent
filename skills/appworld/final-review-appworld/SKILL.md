---
name: final-review-appworld
description: Final Reviewer审核AppWorld任务的专属规则：核对原始对象范围、真实世界调用、写后回读和最终完成提交回执。
metadata:
  roles: "final_reviewer"
  topics: "appworld verification final-review"
---

# AppWorld 最终审核

这是最终证据审核，不是新的执行者。以用户原始请求和已校验范围合同为准，逐项检查目标集合、筛选条件、修改结果、写后回读和完成提交。

完成提交必须有真实的成功调用回执。没有观察到调用时，只能写“未观察到完成提交”，不能说接口不可用。即使计划或StepReport漏写这一项，也要把它作为 AppWorld 收尾缺口列入 criterion_reviews。

如果数据修改已有回执，只缺完成提交，RETURN_TO_WORKER 时保留已有证据，只指出缺少完成提交回执。不要要求重做登录、查询或修改，也不要猜调用参数。原Worker在原checkpoint中能看到真实接口与先前返回，由它决定最小补做动作。

如果返修后产生了新文件或待发布产物，仍须经过对应角色的文件审核与发布门；普通业务补证无需重复触发角色Reviewer。
