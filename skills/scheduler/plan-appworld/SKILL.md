---
name: plan-appworld
description: 用户明确在 AppWorld 模拟环境中测试或操作应用时，安排 General 或 Code 执行并根据真实应用状态验收。
metadata:
  roles: "scheduler"
  topics: "planning appworld"
  conflicts-with: "plan-web-research plan-personal-tasks"
---

# AppWorld 任务规划

当前任务环境无法联网。使用 PLAN 安排 GENERAL 或 CODE。简单查询和直接应用操作交给 General；较复杂的数据处理、可复用脚本或需要独立测试的逻辑交给 Code。Code 保留 Worker、Reviewer、返修与发布流程。

同一应用、同一登录状态下的文档查询、目标分页、修改、回查和 complete_task 应留在一个 GENERAL Step；只有确需切换角色或交付独立产物时才拆分。
GENERAL 先用 appworld_discover 补齐未知接口，再用 appworld_execute 执行与回查；不得为 GENERAL 安排 Code Reviewer 或 appworld_verify。确需 Code Reviewer 独立验证时选择 CODE，并通过现有 Code Worker/Reviewer 流程执行。

Code 脚本使用 code_task 的 ARTIFACT 交付，写明脚本/结果产物、具体要求和可核验标准。发布脚本不等于实际完成应用操作，要求执行者给出 AppWorld 状态证据。Docker 用于本地代码和测试，AppWorld API 必须通过 appworld_execute 访问；不要假设两个环境共享文件或变量。

规划只交代任务目标、对象范围和验收条件；只有当前 Skill、RAG 或真实工具文档明确提供的 API 才可写出名称，未见过的接口绝对不能猜测或举一个臆造名称。缺少接口依据时写“查询真实文档后执行该操作”，不填假方法名。下文明确给出的 complete_task 属于已提供的接口，可以且必须按收尾规则安排。

根据任务需要安排执行者查阅应用 API 文档。Code Reviewer 通过 appworld_verify 独立查询同一世界，不依赖 Worker 临时变量；发现缺陷交回 Worker 修复。依赖操作保持顺序，避免多个角色并发修改同一数据。
文档查询聚焦当前步骤缺的动作与对象，保留用户筛选范围；应用介绍不等于接口签名，不据此编造方法名。

AppWorld 范围案例：用户要求“归档我创建的文件夹中、我标星的文档”。先把“我创建的文件夹内全部文档”的ID记为A，把“我标星的全部文档”的ID记为B，真正目标是A∩B。不能查询“我标星的文件夹”后处理其中所有文档；那会把作用在文档上的条件错误地移到文件夹。计划中明确要求执行者分别取得A和B、按文档ID求交集、只写交集并回读交集；真实接口名仍由执行者查文档。
本例的target_selection应令target_entity=document，A的condition_owner=folder、result_entity=document，B的condition_owner=document、result_entity=document，operation=INTERSECTION，operands=[A,B]，join_key=document_id。不要在该结构中填写API名。
同时按用户动作填写effect_mode：只查询这些文档为READ_ONLY，移动、标记或删除为MUTATION。若最终范围还依赖另一对象中的说明，把它放进required_context，并写清先读什么、为什么以及用于哪一步筛选；不要伪造成同类型集合。

## 关系来源规划

当target_selection.required_context包含relationship时，先规划关系成员的读取与身份关联，再规划最终对象的筛选、修改和回读。关系成员是筛选前提，不是最终操作对象。原样保留Scope给出的source_system、relationship、resolution_status、resolution_note和used_for_sets；不得因为最终对象位于某个应用，就把关系来源自动替换成该应用的好友网络。

- EXPLICIT：按用户明确指定的来源规划。
- DERIVED：保留唯一推导所依赖的已知上下文。
- INFERRED：在execution_guidance中明确要求Worker先核对真实API文档、只读取得关系成员，再通过真实返回中的稳定身份字段关联最终对象；不得把推测描述成用户明说。
- UNKNOWN：先安排关系来源发现，来源未解决前不安排写入。

AppWorld任务族合成英文示例一（不是评测原题）：`Tag incoming Venmo payments from my friends this week.`

该个人关系跨应用筛选任务族使用Phone联系人关系。计划应先取得Phone联系人簿中的friend成员，再按邮箱、电话号码或真实返回中的其他稳定身份字段关联本周收到的Venmo付款，最后只修改匹配付款。可把`phone.search_contacts`作为Skill已提供的候选接口名写入api_suggestion，但不得写参数或签名，Worker仍须用真实文档核验；如果当前版本文档不一致，以真实文档为准并忽略建议。

AppWorld任务族合成英文示例二（不是评测原题）：`Review this month's Venmo transfers with my Venmo friends.`

该平台交易好友任务族使用Venmo好友网络。计划应先取得Venmo好友，再按真实身份字段筛选本月与好友之间的交易。可把`venmo.search_friends`作为Skill已提供的候选接口名写入api_suggestion，但不得写参数或签名，Worker仍须核验真实文档。示例只要求查询，不能凭借本Skill自行添加点赞或写入动作。

这两个例子不构成全局的“friend=Phone”或“friend=Venmo”。先按照Scope已经识别的任务族与关系用途选择；若Phone和目标平台都支持friend且当前材料仍不能消歧，应保留INFERRED或UNKNOWN并让Worker先做只读查证，不得把API存在本身当作语义已经确定。success_criteria必须检查关系来源、身份关联和最终对象集合，而不只是检查已选对象是否操作成功。

**注意：绝对不要忽略最后的完成提交。业务改完、回查通过、打印“成功”、提交 StepReport，都不能代替 complete_task。** 最后一个 Step 的 execution_guidance 必须直接写明：“业务核验通过后，通过 appworld_execute 执行 apis.supervisor.complete_task()，检查真实返回并保留调用证据；未调用或调用失败必须报告未完成。”不要缩写成“最后提交”或“打印成功结果”。success_criteria 另列“完成接口实际调用成功，有工具返回证据”。输出计划前检查这两处均已写入；缺任一处先补齐计划，不把收尾留给执行者猜。按文档选择下面一种形式（answer 只用于任务明确要求的答案，不是总结报告）：

```python
# 操作类任务：确认所有要求的修改均已完成后，再提交完成标记。
print(apis.supervisor.complete_task())
# 问答类任务改用下方形式：actual_answer 必须来自已核验的结果。
# print(apis.supervisor.complete_task(answer=actual_answer))
```

业务未完成则不提交成功标记。Final Review 检查原始要求、执行证据、Reviewer 结论、未完成项及上述收尾证据；完成标记不等于官方评分。只有终止的 Final Review 后外部程序才执行官方评分。无法完成时按现有协议如实失败，不宣称通过。
