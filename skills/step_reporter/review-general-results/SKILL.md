---
name: review-general-results
description: 审核普通 General 步骤的查询、整理或工具操作结果，检查原始范围、真实调用与文件交接；AppWorld、网页检索或专项文件验收优先选对应专属Skill。
metadata:
  roles: "step_reporter"
  topics: "verification"
  exclusive: "true"
---

# General 审核流水线

1. 读原始任务与成功标准，保留标准原文和顺序；Worker总结只是声明。
2. 用tool_audit定位实际操作，再看对应结果。未调用、错误返回、无结果、成功返回分开；成功返回仍需对照要求。
3. 短证据已足够则直接判断；只有关键部分缺失才用read_review_material读取目录中已登记的引用，按next_offset分页。不读取无关目录凑流程。
4. 核查对象范围、必要输出和文件内容；文件存在不等于内容正确。缺证据留UNKNOWN，不替Worker猜原因。
5. 按原标准提交StepReport。批准文件只填approved_artifact_refs，Harness发布后才有共享路径。格式错误只修报告；真实证据不足才提后续核验建议。

## 完整短例：核验文件后批准交接

以下是虚构教学记录，编号以本轮登记表为准，不直接复制为真实证据。

**收到契约。** Step 1/C1：把输入中两条待办整理成一份文件，不增加或漏掉待办。Worker声称已生成；E1包含完整输入，A1是登记候选文件，M1对应其内容。文件存在尚不足以通过。
```python
# 输入证据已经完整，只读取待批准的文件内容。
read_review_material(reference="M1", offset=0)
```
**读取返回。** 内容确实包含输入中的两条待办，没有新增项，next_offset为null。逐条对照E1后，C1通过，批准A1。此时文件仍是候选，不能声称已进入共享目录。
**提交结论。** C1＝MET，引用E1/A1；整体COMPLETED，说明两条待办均保留且没有新增项。批准A1，artifacts留空，按当前提交工具的schema填写报告。
Harness随后处理发布并提供实际交接路径；Reviewer到提交即结束，不自己写入共享目录。若读取返回UNAVAILABLE，只能报告内容未核验，不批准A1，不能照抄本例的通过结论。
