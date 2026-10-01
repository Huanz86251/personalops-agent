# PersonalOps Agent 最终面试演示目标

这份文档定义最终应该展示什么，不把尚未实现的内容冒充完成结果。

## 面试官看到的五层证据

1. Evaluation 总览：固定模型、任务清单、预算、Git SHA、工具 schema hash、
   Skill hash；展示 baseline 与 optimized 的官方成功率、Token、工具错误和延迟。
2. Phoenix 单题 Trace：PREPARE、RUN、GRADE、SAVE；RUN 内展开 planner、
   executor、LLM 和工具错误恢复；官方评分与 Agent 自报分开。
3. SkillOpt 优化过程：失败轨迹如何产生 add/delete/replace Skill patch，候选
   如何通过 held-out validation gate，失败候选如何被拒绝。
4. Skill 管理：每个 Skill 的用途、版本、来源、适用任务、验证成绩、当前
   active/best 版本；运行时只注入被路由选中的 Skill。
5. Event 运行时：用户连续输入、优先级、暂停/恢复/取消和 Trace 关联。它需要
   独立事件一致性 Evaluation，不能用普通 AppWorld 成绩代替。

## AppWorld 在闭环中的位置

AppWorld 是带初始数据库状态的模拟应用世界。Agent 接收自然语言任务，通过
appworld_execute 写代码调用模拟 API。官方 grader 在 finish 后检查最终状态和
副作用。因此它能验证：

- 多步规划和跨应用操作；
- API 文档发现与参数遵守；
- 状态保持、循环、过滤、聚合等代码能力；
- 工具错误恢复和重规划；
- 是否真正完成目标；
- 是否误改无关数据。

一个 appworld_execute 是一个通用执行入口，不等于只有一种能力。它内部可访问
AppWorld 记录的应用 API。这个设计测的是 Agent 是否能自主发现、组合和验证
API，而不是背固定工具列表。

AppWorld 不能单独证明 PersonalOps 的个人记忆、RAG、真实账户集成、并发事件
调度或长期对话能力；这些需要各自的 Evaluation。

## SkillOpt 如何接入

SkillOpt 不训练模型权重。目标模型、Agent 图、工具和预算在对照中冻结，Skill
Markdown 是可优化的文本参数。

每一步：

1. 目标 Agent 使用 current_skill 跑 Train 小批任务。
2. AppWorld grader 返回 hard score；诊断器生成工具错误、Token、预算等 soft
   signals；Phoenix/私有轨迹提供失败过程。
3. optimizer 模型分析多条成功/失败轨迹，提出有界 add/delete/replace patch。
4. candidate_skill 在固定 validation 任务上重新运行。
5. 只有 validation 严格改善才晋升为 best_skill.md，否则记录拒绝。
6. 最终只在冻结 Test 上跑 aggregate score，不看逐题答案做优化。

不得把题目答案、固定 API 调用、具体模拟账号或 test 信息写进 Skill。可回流到
真实 PersonalOps 的内容应是通用策略，例如先查 schema、使用关键字参数、合并
依赖操作、变更后验证、失败分类和停止条件。

## 要展示的指标

主指标：

- 官方任务成功率：一题的官方测试全部通过才算成功。
- paired delta：相同任务 baseline 与 best_skill 的成功差。
- 置信区间或至少逐题 paired 表，不用两个单题制造 0 到 100 的说法。

诊断指标：

- 工具调用数和明确错误率；
- unknown API、参数校验、被禁模块等错误分类；
- 模型调用数、完整 Token、未知 usage 数；
- 延迟、预算触顶和截断；
- Agent 自报与官方 grader 不一致次数；
- Skill 长度和 inference-time 额外模型调用。SkillOpt 的部署 Skill 本身不应
  增加 optimizer 调用。

## 现场点击顺序

1. 点击 interview_demo/02_只打开Phoenix查看已有结果.py，先展示稳定的已有证据。
2. Phoenix 中打开一条真实 baseline Trace，解释一次失败。
3. 打开 SkillOpt dashboard 或本地实验报告，展示 candidate patch 与 gate。
4. 展示 baseline vs best_skill 的固定任务 paired 表。
5. 打开 best_skill.md diff，说明哪些规则来自哪些失败簇。
6. 打开 best_skill 的新 Trace，展示工具错误减少或更早完成。
7. 最后展示 Event 架构图和独立测试结果，不把尚未完成的异步功能说成已上线。

真实运行可点击 interview_demo/03_运行真实Agent评测_会调用模型.py；它固定单题
和调用上限。批量 baseline 与 SkillOpt 训练必须使用冻结 manifest，不放进现场
一键脚本，避免误触大量费用或污染实验。
