> 当前恢复入口与停止边界见 [SESSION_HANDOFF.md](SESSION_HANDOFF.md)。用户要求今天接通与验收后停止，后续优化须结合共同学习推进。

# PersonalOps：AppWorld 与代码执行型个人助手评测路线

更新时间：2026-08-31。
状态：公开资料调研与设计建议；未运行基准、未接入 SkillOpt、未产生提升数字。
用户目标：约十几天内准备 Agent 应用开发面试；保留个人事务助手定位，重视代码执行、SkillOpt 和工具改进；不做模型权重训练，文档 RAG 暂缓。
本版替代此前电商售后建议。电商与用户的产品定位不一致，不再作为主攻路线。

## 1. 当前首选

**AppWorld 作为公开主基准；将 PersonalOps 做成用代码和工具完成个人事务的 Agent；SkillOpt 优化其通用操作技能。**

不是另造名为“AppWorld-CodeAct”的基准，也不是重新训练底层模型。代码执行型 Agent 指它可以写一段代码，执行、观察输出或错误，再继续执行下一步，而不是只返回代码文本。

如果最终明确转向开发者终端助手，则改选 Terminal-Bench；只有要做“修复真实代码仓库 issue”的产品时才优先 SWE-bench。十几天内不同时主攻三个。

## 2. Benchmark 与 framework 的区别

GSM8K 属于任务基准：规定题目及正确性。Agent 的对应物是 AppWorld、Terminal-Bench、SWE-bench、BFCL 等。框架负责运行这些题、管理环境并收集成绩；观测工具负责追踪 Token、延迟和错误。

| 工具 | 类型与用途 | 本项目的决定 |
| --- | --- | --- |
| AppWorld Engine + evaluator | 模拟应用、任务与结果验收 | 首轮直接接其运行与评分接口 |
| Harbor | 容器中的 Agent 评测框架，可接自定义 Agent 和多个基准 | 走 Terminal-Bench 时优先采用；不为统一界面强行迁移 AppWorld |
| Inspect AI | 支持工具、多轮交互与模型评分的通用评测框架 | 备选，不是自带一个通用百分制成绩 |
| DeepEval | Python 评测库，含任务完成、工具及其他指标 | 补充回归，不替代公开基准官方评分 |
| Phoenix | 观测、数据集和实验 | 复用当前项目已有接入 |

来源：[Harbor](https://www.harborframework.com/docs/core-concepts)、[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)、[DeepEval](https://github.com/confident-ai/deepeval)、[Phoenix](https://arize.com/docs/phoenix/evaluation/evals)。

“使用 DeepEval 得到 90 分”如果没有交代任务与指标，无法与他人比较。“固定模型和预算，在 AppWorld Test-N 上 TGC 为某值”才是标准化得多的成果表达。

## 3. 候选基准到底测什么

| 基准 | 一题的形态 | 典型成绩 | 与本项目关系 |
| --- | --- | --- | --- |
| AppWorld | 在模拟个人应用中查信息、写代码、调 API、完成事务 | TGC、SGC | 首选；个人事务与代码执行同时保留 |
| Terminal-Bench | 在隔离终端环境完成编程、调试、数据或系统任务 | 任务成功率，固定版本/超时口径 | 若重点是终端自动化则更直接 |
| SWE-bench Verified | 给真实仓库与 issue，生成修复补丁并通过测试 | resolved rate | 适合专门的软件修复 Agent |
| BFCL | 选择函数、填参数、多步/多轮工具操作 | 分类准确率及官方综合分 | 工具诊断可补充，不能替代完整产品评测 |
| GAIA | 工具辅助的研究与问题解答，部分题有附件 | 答案准确率 | 偏检索与问答，非当前写代码办事务的首选 |
| HumanEval+ / MBPP+ | 实现给定编程功能，用增强测试验收 | pass@1 等 | 适合作为基础代码能力检查，较难体现完整 Agent 系统 |

来源：[AppWorld 论文](https://arxiv.org/abs/2407.18901)、[Terminal-Bench 2.0 论文](https://arxiv.org/abs/2601.11868)、[SWE-bench 数据说明](https://www.swebench.com/SWE-bench/guides/datasets/)、[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)、[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)、[EvalPlus](https://evalplus.github.io/)。

这些基准有公开论文、执行/评分代码或榜单，提供可追溯依据，但没有哪个统一代表全部 Agent 能力。“认可度”不应变成保证面试官一定认可某个数字。

版本注意：Terminal-Bench 官方主页目前展示后续版本，2.0 是一个有论文和固定任务集的既有版本；不把 2.0 称为最新版，不能跨版本比较分数。SWE-bench Verified 也要固定数据与 harness，不能与其他 SWE 变体混用。

## 4. 为什么 AppWorld 合适

AppWorld 本身就是面向 function calling 与 interactive coding 的应用环境。它提供 9 个日常应用、457 个 API，以及标准任务和评分器，不要求真实用户账户；当前还提供 MCP 接口。初期可以从其代码执行接口切入，不必一口气向模型暴露所有 API。[官方仓库](https://github.com/StonyBrookNLP/appworld)

其中代码是执行任务的手段：Agent 根据文档查询记录、筛选/汇总数据、调用合适 API，检查变化。它并不是 SWE-bench 那样主要提交对已有项目的代码补丁。

以下是自拟示意，不引用测试题：“根据笔记里的运动时长，从我的收藏中选择合适的歌单并播放。”可观察的工作包括读笔记、查歌单、计算时长、发起播放和验证结果；代码适合承担遍历和计算。所有操作只在模拟世界进行。

标准指标：
- TGC（Task Goal Completion）：完成全部检查的任务占比。
- SGC（Scenario Goal Completion）：同一场景的不同任务变体全部成功的场景占比。
SGC 不是同一题随机重跑三次成功率。后者可以另外测，但不能混名。[指标定义](https://arxiv.org/html/2407.18901v1)

原论文的划分为 Train 105、Dev 60、Test-N 168、Test-C 417；这是原始基准规格，实际安装后核对固定版本的清单。Test-C 涉及训练/开发未出现的指定应用，衡量更强的泛化；首轮标准成果目标为 Test-N 全集。不得针对 Test-C 预先编写专用解题 Skill。[原论文数据划分](https://arxiv.org/html/2407.18901v1)

不存在“达到 50% 就一定很好”的统一阈值。分数要和相同模型、预算、版本、工具接口下的公平基线比较；不能把旧论文某个模型的分数当今天的直接对照。

## 5. 一题怎样跑，batch 又是什么

一次完整 trial：
1. 控制程序从指定 split 取题，初始化该题的模拟世界。
2. 向 PersonalOps 提供任务、允许的应用/API 文档入口及工具；不提供答案或验收代码。
3. 现有规划/执行链加载固定版本 Skill，决定下一步。
4. Agent 生成代码或工具调用，在该题的隔离环境执行。
5. 将输出/异常回给 Agent；在固定次数/时间/Token 预算内继续。
6. Agent 结束后，由外部官方 evaluator 检查结果。
7. 保存运行配置与使用量；训练/开发允许分析逐题报告，正式 test 按官方规则只看汇总。

batch 只是对一份任务清单重复上述过程。比如 10 个开发案例，是 10 个独立的初始世界、10 次 Agent 运行和 10 次评分，不是把 10 题塞进一次模型请求，也不是模型训练的梯度 batch。

先串行跑正确，再加独立工作进程；每题和重复试验各自拥有环境、会话、记忆及输出路径。AppWorld 官方明确单个环境服务进程同一时刻只能持有一个 world；同进程 async/thread 并发多个 world 不符合该隔离要求。[官方并行指南](https://github.com/StonyBrookNLP/appworld/blob/main/guides/parallelizing_worlds.md)

一个实验组得到 TGC/SGC 与成本报告；更换 Skill 或工具说明后用同样协议再跑，形成对照。并发用来缩短耗时，不是虚增样本或提升任务成功率。

## 6. 现有代码要怎样接

已核对存在 conversation_runtime、planning_graph、agent、toolsets、ShellToolMiddleware、Playwright MCP 和 Phoenix。它们是可复用基础，但不等于已经满足基准协议。

需要补：
- 离线运行入口：评测时任务来自数据集，不需要逐条从飞书发送。
- 环境适配器：将执行工具接到 AppWorld 任务世界，而非宿主 Windows Shell。
- Skill 加载/版本记录：指定本轮运行的技能，不使用跨实验变化的隐式状态。
- 状态隔离：避免默认 owner 会话与 SQLite 记忆串题。
- 实验日志：数据/模型/Skill/工具/配置版本、Token、成本和结束原因。
- 独立评分调用：由 runner 在 Agent 结束后触发，测试期间不可把分数或目标状态反馈给 Agent。

首轮采用代码执行模式：代码调用公开应用 API；保留现有 planner/executor。MCP 工具调用模式可以作为后续不同执行方式的比较，不混在同一个“仅改 Skill”的实验里。

AppWorld 当前要求 Python 3.11+，项目说明的开发环境为 Python 3.10。优先独立环境/服务，避免直接升级并破坏现有依赖；兼容性尚未实测。官方的终端 Agent 接入说明也表明代码工具和终端 Agent 可以接入 AppWorld。[终端 Agent 接入](https://github.com/StonyBrookNLP/appworld/blob/main/guides/evaluating_terminal_agents.md)

安全必须由进程/容器和权限约束保证：AppWorld 默认同进程执行不等于强沙箱。不挂载真实私人目录、密钥、答案或数据库管理接口；Agent 不得直接读取/修改后台 DB，也不能调用评分器“探答案”。

## 7. SkillOpt 优化什么

计划中的通用 Skill 内容可涵盖：
- 根据任务发现并阅读 API 文档；
- 先检查数据结构和分页，再汇总；
- 用少量代码完成数据筛选/计算；
- 执行写操作前核实对象、范围与用户约束；
- 依据异常修正调用，有界重试，避免重复副作用；
- 用允许的查询验证结果，再报告完成；
- 控制长输出和中间状态。

这是候选设计，不预先断言这些规则一定提高分数；按 Train/Dev 失败分布迭代。禁止硬编码特定题目的账户 ID、答案、调用序列，不能提前获取 Test-C 隐藏应用任务知识。

SkillOpt 固定任务模型，根据执行轨迹提出有界的技能文本修改，在验证集上选择候选。要实现正式 benchmark adapter，将完整的 PersonalOps-AppWorld rollout 与官方评分接入，而不是另写一个简单模型问答器来代替项目。[SkillOpt 仓库](https://github.com/microsoft/SkillOpt)、[扩展接口](https://github.com/microsoft/SkillOpt/blob/main/docs/guide/new-benchmark.md)

这不是现成一键适配：AppWorld 与 SkillOpt 的版本、执行后端及数据接口需小样本联调。使用 DeepSeek 时也要验证选定 SkillOpt 版本的兼容后端。

## 8. 对照组、数据协议与工具改进

| 组 | Skill | 工具/API 说明 |
| --- | --- | --- |
| A | 只有完成基准所必需的通用说明 | 官方原始说明 |
| B | 人工初版 Skill | 官方原始说明 |
| C | SkillOpt 优化版 | 官方原始说明 |
| D | 同 B | 受控改进说明 |
| E | 同 C | 同 D |

各组使用同一任务模型、Agent、预算和环境。C-B 测自动技能优化，D-B 测工具说明，E-B 测联合效果；不能只展示最弱 A 与最佳 E。A 仍有完整必要文档与工具，不故意限制以制造提升。

首轮以 A/B/C 为主，D/E 作为有余量的扩展。改工具描述与改真实函数签名是两类工作；后者需要适配与独立验证。SkillOpt 不自动等于通用 schema 优化器。

AppWorld 官方协议：
- Train 用于学习、示例与错误分析；本项目用它优化 Skill。
- Dev 用于调参、分析与候选选择。
- Test-N/Test-C 仅用于最终测试及汇总；开发者不查看题目/逐题评测报告做错误分析或改提示词。
- 允许从 Train/Dev 总结通用提示，不允许在 Agent 代码中硬编码业务 API 调用去代替自主决策。
遵守这些规则比“我们本地把数据叫 test”更重要。[官方开发限制](https://github.com/StonyBrookNLP/appworld#no_entry_sign-agent-development-restrictions)

测试运行时 Agent 接收本题是必要行为；上面的禁止是开发者/优化器不能利用测试内容调参。测试轨迹与答案应隔离保存，报告只展示允许的汇总；面试失败案例从 Train/Dev 取。

若自选题或改任务/接口/描述，标为相应子集或受控变体，并核实官方提交协议；不默认等价于官方榜单分数。SGC 需要完整场景组，不能任意抽单题后冒充标准 SGC。

成本统计包含全链路模型调用、摘要和失败重试，优化器费用单列；服务商 usage 缺失标为估算，不填零。保留错误和超时，记录模型/数据/代码版本和限制。改善率及百分点区分，小样本报告统计不确定性。

## 9. 十几天的具体目标

第 1—2 天：独立环境，校验官方训练/开发环境可运行，接入真实 PersonalOps，跑 5—10 个 Train/Dev 案例估算成本。
第 3—4 天：稳定代码执行循环、隔离状态、形成 A/B 开发基线。
第 5—8 天：接入 SkillOpt，自训练集优化，在 Dev 选择；记录技能变化与失败分类。
第 9—10 天：冻结主要方案；有余量做工具说明/Token 优化，所有选择仍在 Dev。
最后数天：统一冻结多个比较组，运行 Test-N；展示 TGC、SGC、Token/成本与限制。Test-C 视预算作为泛化扩展，不提前调它。

日程为建议，不是兼容性或工期承诺。全量之前预算 = 案例 × 实验组 × 重复数 × 实测平均成本，另加优化循环花费。预算不足时如实报告固定子集，不能写成全集成绩。

不同时接入 Terminal-Bench、SWE-bench 和 AppWorld，不以“加入更多框架”替代拿出可复现结果。

## 10. 成果如何表述

实测之后可写：
“构建基于代码执行与工具调用的个人事务 Agent，在固定 [模型/版本/预算] 下，于 AppWorld Test-N [全集或明确子集] 取得 TGC [A]、SGC [B]；采用 SkillOpt 后，相对同模型同配置基线，TGC 变化 [C] 个百分点，平均任务成本变化 [D]。”

这只是字段模板，没有实际数字。不能承诺 50% 或其他分数一定达到，也不能把 Agent 系统得分说成底层模型的裸能力。

面试主线：如何把自然语言任务变成可执行代码/API 操作；如何保证执行与数据隔离；如何设计和优化可复用 Skill；怎样用标准任务与独立评分证明收益；怎样在质量、可靠性和成本间做选择。

