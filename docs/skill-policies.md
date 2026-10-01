# Skill 策略正文与实际执行流程

> 当前状态已由同日后续 [38 项场景 Skill](skill-scenarios.md) 更新，全部采用新名称且不保留兼容别名；[真实角色演练](skill-role-field-test.md) 记录Docker交接与测试结果。下文的12项、旧名称兼容、1,750-token Web正文和旧目录总量是前轮历史记录，不是当前配置；公开参考与取舍仍保留。

更新：2026-09-06。用户希望 Skill 提供能指导决策的 policy，而非一句通用口号。本轮重写原有 8 项，新增 4 项，共 12 项；保留已有名称兼容 fixed 配置。

同日后续：Web source-research 已补充中文实体查询、按领域选源和登录/动态页/地图回退。来源核对、项目真实文本读取抽样与局限见 [Web 策略工作记录](worklogs/2026-09-06-web-source-policy.md)。该调整后的 Web 正文约 1,750 token，下面 9,135-token 总量统计属于此前版本，不能当当前目录总量。

## 原方案与本轮修正

9 月 5 日的方案是角色内准备：Scheduler 先选方法再正式规划；各 Worker 与 Code Reviewer 独立准备；方法正文冻结到角色快照；固定角色协议负责工具、权限、提交与验收。后续 Prompt 压缩把 Skill 正文也缩到一句话，保留了接线，却丢失了方法本身的判断步骤。本轮修正这一内容问题。

这里的 policy 是可复用的执行策略：触发条件、观察依据、行动分支、验证方法及收敛条件。它可以提高模型遇到具体情况时选对下一步的机会；实际收益需通过任务行为评估，不能根据文本变长或测试接线通过断言。

固定角色 Prompt 保留始终有效的职责和提交协议。Skill 只承载按任务选用的方法。例如“Reviewer 不修改产品代码”属于角色边界；“状态任务重开临时存储验证恢复位置”属于可选策略。

## 公开参考与取舍

以下一手页面于 2026-09-06 查阅。采用其组织方式与方法思想，正文按 PersonalOps 契约、工具和预算独立编写；没有复制整个第三方 Skill 包或安装插件。PR Review Toolkit 中两份材料是 Agent 指令模板，并非 SKILL.md，借鉴的是审核方法。

| 参考 | 有用之处 | 本项目的改写 |
| --- | --- | --- |
| [Agent Skills 格式规范](https://agentskills.io/specification) | 简介用于发现，正文包含步骤、例子与边界，资源可逐层披露 | 简介写清适用条件；完整策略仍在可冻结的 SKILL.md |
| [Superpowers systematic-debugging](https://github.com/obra/superpowers/blob/main/skills/systematic-debugging/SKILL.md) | 从复现与因果调查进入修复，实验用于区分假设 | 增加 root-cause-debugging；服从现有预算，不引入固定三次失败就要求人工架构讨论的通用门槛 |
| [Superpowers test-driven-development](https://github.com/obra/superpowers/blob/main/skills/test-driven-development/SKILL.md) | 测试应能暴露目标行为缺失，以可观察结果证明实现 | 在有逻辑变更时优先回归检查；不照搬删除已有实现、任何小修改都强制 TDD 的规则 |
| [Anthropic code-reviewer](https://github.com/anthropics/claude-plugins-official/blob/main/plugins/pr-review-toolkit/agents/code-reviewer.md) | 聚焦实际影响、文件位置和有依据的缺陷 | 以 requirement ID、触发输入和结果支持返修；不引入没有校准依据的数字置信阈值 |
| [Anthropic pr-test-analyzer](https://github.com/anthropics/claude-plugins-official/blob/main/plugins/pr-review-toolkit/agents/pr-test-analyzer.md) | 按行为、错误路径和重要缺口评估测试质量 | Reviewer 使用按风险选测试的表；只覆盖契约相关分支，不追求任意 100% 覆盖率 |
| [Anthropic webapp-testing](https://github.com/anthropics/skills/blob/main/skills/webapp-testing/SKILL.md) | 先观察已渲染状态，再使用实际找到的控件行动 | Web 方法采用观察—行动—核对；不要求当前环境没有的辅助脚本或 Playwright，不将 networkidle 当通用就绪条件 |

## 当前目录与触发区别

| 角色 | Skill | 关键策略 |
| --- | --- | --- |
| Scheduler | task-decomposition | 按产出与依赖拆分，明确交接和验收，简单任务保持单步 |
| Scheduler | recovery-planning | 按失败层级选择 CONTINUE / RESTART / STOP，复用有效成果 |
| Scheduler | uncertainty-driven-planning（新增） | 优先调查会改变方案的未知项，为调查设置决策分支和停止条件 |
| Scheduler / Step Reporter | evidence-synthesis | 逐项标准与证据对应，消解版本/口径冲突并显式保留缺口 |
| Code | incremental-implementation | 契约映射、调用链定位、最小纵向实现、针对性验证与候选提交 |
| Code | root-cause-debugging（新增） | 最小复现、定位错误边界、单一假设实验、局部修复与防回归 |
| Reviewer | behavioral-verification | 独立推导测试、风险矩阵、分类失败、具体返修、按版本复验 |
| Reviewer | state-transition-verification（新增） | 按需检查重复、乱序、失败、重开存储、取消及并发不变量 |
| General / Web | document-reading | 目标定位、分页、表格口径、OCR 限制和已读范围 |
| General | structured-comparison（新增） | 按统一维度综合已有材料，区分未知/不支持，有条件地给建议 |
| Web | source-research | 从证据缺口检索一手材料，核对时间与冲突，证据足够即结束 |
| AppWorld General | appworld-api-workflow | 查真实接口、定位 ID、有限变更、写入不明先回读、逐项验收 |

新增 Skill 位于现有扫描目录，无需更改 sources。一般角色不必同时选择所有方法；例如清晰的新功能通常选 incremental-implementation，纯故障定位优先 root-cause-debugging。状态相关审核可以将两个 Reviewer 方法组合。

skill_selector 增加互补性选择与短例子；最大仍为 3，可以选空。示例中的 ID 只有实际存在于候选时才可选。候选按 role、source/topic、required-tools 过滤，Skill 不授予新工具。

## 实际调用顺序

1. `planning_graph.prepare_scheduler` 扫描并冻结目录。给选择模型的仅为候选 ID / description 和任务上下文，不发送全部正文。
2. Scheduler 的选中正文进入固定 system 前缀；正式规划仍收到原始任务。Step 选择 Worker 类型，并可提供最多三个 skill_topics。
3. Worker 图中的 `RoleSkillsMiddleware.before_agent` 用该角色的模型选择自己的方法，再把选中正文加入执行 system。`code_runtime` 把同一份目录、模式、topics 与 code_task 传入 Code 两个角色，但 Worker 与 Reviewer 各自创建选择快照。
4. Code Worker 完成修改与自检，沿当前提交协议交付候选。Reviewer 得到 code_task、候选与结构化提交清单，按独立方法设计检查；不会获得 Worker 的选择聊天记录或隐藏推理。
5. 确认缺陷后 Reviewer 提交 bounded repair；Worker 返修后 Reviewer 核对当前版本并复验。审核通过后的发布和报告仍由现有工具/状态机控制。
6. 普通返修、Scheduler CONTINUE、checkpoint 恢复沿用各自快照。新 run 才采用本轮更新的正文；本轮没有增加 run 中途重选、磁盘热更新或辅助文件读取机制。

当前所有方法保持自包含，因为辅助 resources 的冻结和角色访问机制尚未接入。把步骤拆到不可读取的 references 会让它们在真实运行中缺失，因此本轮不这样拆。各 Skill 内用条件分支省略无关工作；条件段一旦随 Skill 加载仍占输入 token，这与真正分段懒加载不同。

## 场景走读：希望策略改变什么

这些是人工静态走读与后续评估样例，不是模型运行成绩。

| 请求/候选 | 应触发的方法 | 希望观察到的行为 | 不足的证据 |
| --- | --- | --- | --- |
| 新增筛选导出 | incremental-implementation | 构造匹配/不匹配项，从真实入口导出并回读内容 | 只断言文件存在 |
| 解析器回归使结果变空 | root-cause-debugging | 比较原始输入、解析和过滤结果，定位首个错误转换 | 在调用方硬编码预期数据 |
| 审核“坏记录跳过，好记录继续” | behavioral-verification | 混合输入，核对保留记录、非法项处理及产物可读性 | 只重跑 Worker 的单个正常用例 |
| 审核事件恢复后不重复创建 | behavioral-verification + state-transition-verification | 关闭并重开临时存储，核对恢复位置与实际副作用次数 | 对同一个内存对象连续调用两次 |
| 附件格式不明，需按月汇总 | uncertainty-driven-planning | 小样本读取后按表格/OCR/缺字段分支安排执行 | 未读样本就直接安排全量计算 |
| 比较计费周期不同的两个方案 | structured-comparison | 按用户用量统一口径，保留未知项和适用条件 | 直接按页面标价排名 |

## 验证与效果边界

本轮 12 项 skill-creator 格式校验通过；`tests.test_skill_preparation`、`tests.test_code_agents`、`tests.test_code_review_models` 共 24 项测试通过。新增检查覆盖各新增方法的角色/工具候选范围，以及两个组合策略的实际中间件注入和恢复后复用；不同角色与未选中正文不进入执行 system。其余检查覆盖已有 Code 提交、返修和发布协议。

以 cl100k_base 统一估算：12 项正文合计 9,135 token；单项为 586–1,324 token；两项 Reviewer 策略连同注入引导约 2,249 token。完整目录的 ID/description JSON 约 904 token，实际选择前仍按角色与工具过滤。总目录正文不会全部发给单个执行角色。此统计不代表服务商分词、实际缓存命中或账单；所有条件分支随选中 Skill 一并加载。

格式校验只证明可解析；目录、候选过滤、正文注入和 checkpoint 测试证明接线。脚本模型按预设 ID 返回，不能证明真实模型会选对，也不能证明它会遵守正文或更容易解题。

正式对照需要固定任务、模型、工具、预算和角色协议，分别观察选择适配度、测试是否能捕获候选缺陷、返修是否有证据、总调用与 token、最终任务结果。不能拿本轮提示词变化直接与旧 AppWorld baseline 做归因比较。此次没有启动付费模型、AppWorld 任务或 SkillOpt。
