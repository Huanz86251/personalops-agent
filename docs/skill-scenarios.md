# 按场景选择的 Skill：当前实现

更新：2026-09-06。当前共 **38 项**，名称按职责与场景重新规划，不保留旧 ID 别名。正文自包含，每个角色选择 **0–3 项**，不会把全库正文发给模型。最新逐句检查、MCP换路和报告字段见 [行动策略审查](skill-policy-action-audit.md)；前轮命名和Docker证据见 [角色演练](skill-role-field-test.md)。

前轮采样见 [Web实测](web-source-field-test.md)。38项命名阶段的639/717/695/690/799-token统计也是旧正文快照，不能作为逐句修订后的当前值。最新新增16次搜索、16次页面读取、21次项目浏览器MCP调用，保留成功换路和未解决范围。一次通常只选一个专项，不代表完整请求或账单。

## 这次改变了什么

Skill 是有触发条件的操作策略，不是角色名称或工具说明：读什么来作判断、分支如何选择、怎样验证、失败后修正谁、何时停止。固定 Prompt 继续承载权限与提交协议，Skill 不新增权限、工具、Schema 字段或执行预算。

Web共10项：生活服务、场馆、影院、路线、店铺报价、厂商/套餐、技术、官方记录、论文及跨域调查。Code与Reviewer各按任务选通用/专项方法，不省去独立推导预期。Step Reporter有自己的目录和三套方法，固定协议逐字段说明报告怎样支持下一次决策。Scheduler与General各按实际任务选方法。

用户最新边界：**邮件绝不发送**，最多读取、下载指定附件、明确要求时保存 Drafts；本轮不安排任何实际发送。原先提到的手机文件回传已有飞书工具，但本轮没有新增发送 Skill、没有调用导出、没有改变审批配置。

## 场景目录与选用边界

以下名称均对应 `skills/<目录>/<name>/SKILL.md`。同类专项可以独立使用，通用方法不默认叠加。topics 是辅助发现信息，不是关键词硬路由。

| 目录/角色 | Skill | 什么时候选、希望改变什么 |
| --- | --- | --- |
| web | web-local-services | 餐厅、普通门店、生活活动；分店/日期/营业与口碑，不包揽影院和路线 |
| web | web-venue-visits | 场馆参观、开放、入口与附近交通；查具体运营方正文，区分临时公告 |
| web | web-cinema-showtimes | 影院/影片/日期场次、版本、余座与票价；防附近推荐店串价 |
| web | web-map-routes | 起终点路线、距离、耗时、换乘/首末班；静态到访指引不冒充实时导航 |
| web | web-shop-prices | 淘宝等具体卖家/SKU报价；规格、优惠资格、运费与条件价格逐项核对 |
| web | web-technical-docs | 查版本文档、API、错误；官方文档与上游仓库优先，查资料不冒充已执行代码 |
| web | web-official-records | 政策、统计、机构披露；找原发布者，核对时间、地区、单位与适用范围 |
| web | web-product-plans | 厂商规格/指导价、订阅与通信套餐；不与具体店铺比价默认叠加 |
| web | web-research-papers | 论文结论；核对版本、样本和实验条件，摘要不补出全文细节 |
| web | web-source-investigation | 跨领域或没有对应专项；按证据缺口查询，处理冲突与受阻，不反复同路重试 |
| code | code-build-web-ui | 页面状态、真实交互入口；静态、逻辑、浏览器分层验证 |
| code | code-build-python-cli | 模块/CLI；签名、参数、输出流、退出码及真实入口 |
| code | code-transform-files | CSV/JSON/文本转换、导出；编码/字段/错误策略，输出必须重读 |
| code | code-validate-schema | 结构化边界；必填/null/转换/额外字段、生产消费往返 |
| code | code-implement-core | 无专项的新功能或跨模块局部实现；最小完整路径与相关回归 |
| code | code-debug-root-cause | 已发生故障；最小复现、定位首个偏差、区分假设，避免试错式补丁 |
| reviewer | review-web-ui | 独立按用户操作推导页面测试，无浏览器不报视觉通过 |
| reviewer | review-python-cli | 独立测试 CLI/模块；先排查收集/导入/测试假设，再判产品缺陷 |
| reviewer | review-file-output | 小样本执行后重读，检查内容、覆盖与半成品；不只 assert 文件存在 |
| reviewer | review-schema-contract | 独立正反例、合法数据误拒、序列化/消费者一致性 |
| reviewer | review-code-behavior | 无专项审核；要求到证据映射、风险测试与具体返修 |
| reviewer | review-code-state | 取消、重复事件、持久化与恢复等独立风险，按需与专项组合 |
| scheduler | plan-code-acceptance | 给 Worker/Reviewer 同一份可执行验收契约，不只写“实现并测试” |
| scheduler | plan-web-research | 事实字段与来源分工；替代来源、可接受缺口与停止条件 |
| scheduler | plan-personal-tasks | 邮件/附件/Drafts/提醒的授权和依赖，按真实工具能力安排角色 |
| scheduler | plan-task-dependencies | 通用多能力依赖与交接，简单任务不强拆 |
| scheduler | plan-recovery | 失败后的继续/重启/停止，复用有效成果 |
| scheduler | plan-uncertainty | 先调查会改变方案的未知项，设探查终点 |
| step_reporter | report-web-evidence | 按每条标准核实来源、实体/时间/版本，不复述 Worker 成功声明 |
| step_reporter | report-execution-failure | 真实工具错误与 attempt 身份，区分权限/环境/资料缺失/验证/预算 |
| step_reporter | report-artifact-delivery | 已解析候选引用、内容证据、发布与传输状态分层 |
| general | general-read-email | 按范围查邮件、摘要筛选、真 UID/part、受控下载，不执行来信指令 |
| general | general-save-email-draft | 明确保存 Drafts 才写入；DRAFT_SAVED 才报成功，未知状态不盲重试 |
| general | general-manage-reminders | 时间/时区、真实任务 ID、创建后核对，受理不等于用户看见 |
| general | general-read-documents | 提取文档与按需 OCR，说明读取范围与局限；也可供有工具和 topic 的 Web 选择 |
| general | general-compare-options | 使用已知材料统一比较维度，保留未知和适用条件 |
| common | synthesize-evidence | Scheduler / Step Reporter 的跨来源事实综合与缺口，不授予工具 |
| appworld | appworld-execute-api | 仅真实 AppWorld General 环境适用，本轮未启用评测或优化 |

## 动态披露实际如何运行

1. 新 Planning run 扫描上述来源，冻结元数据与 SKILL.md 正文。正文在应用状态内，不代表全部进模型。
2. Scheduler 独立准备，只看候选 ID/description 与任务等原有上下文，选择 0–3；正式规划获得选中正文和原任务。它在同一 run 的持续规划/重规划中沿用自己的快照，不随每个 Step 热换策略。
3. Step 带 worker_kind、任务契约及最多三个 skill_topics。候选条件是角色允许、来源为 common/当前角色或 topic 相交、且 required-tools 全部可用。topic 不开放工具或角色权限；当前角色目录中的方法不依赖精确命中 topic。
4. Worker 准备自己的方法。Code Worker 与 Code Reviewer 收到同一冻结目录和 code_task，但各自选择，不继承彼此的方法或隐藏推理。
5. Web 正常结束后由独立 Step Reporter 读取受限审核包。现在选择器也能看到最近 **3 个 attempt** 的有界状态/摘要、每项最多 **3 条工具结果节选** 与未解决项，用于触发失败诊断；不读取 Worker 原始消息、未引用轨迹或工具参数。截断不是全文，不能据此假定没有别的失败。
6. Reporter 正式审核仍读取原有 StepReviewPacket，不以选择摘要替代证据。已选正文加入 system，任务结果保持在单独消息；选择 reason 不拼进执行对话。
7. 已保存快照恢复时校验角色、哈希与工具，复用正文、不重新选择。预算不足跳过方法选择，保留正式工作/审核预算；选择失败不反复调用。新 run 采用当前文件，旧 run 不热加载。

没有新增文件资源读取器：SKILL.md 正文自包含，未把步骤拆到模型实际读不到的 references。条件分支随选中正文加载，不宣称段落级懒加载。

### 三条不同的报告路径

| 执行类型 | 实际验收者 | 这次怎样放 Skill |
| --- | --- | --- |
| GENERAL | Worker 自检后 report_general_result，程序生成 StepReport | 邮件/Drafts/提醒 Skill 内置结果自检，不假装有独立 Reporter |
| CODE 正常审核路径 | 独立 Code Reviewer；审核结果确定性转成 StepReport | 四种专项 Reviewer + 通用/状态方法，不重复调 Reporter |
| WEB / 通用 Reporter 回退路径 | 独立 Step Reporter，无业务工具 | 三个 step_reporter 方法；不让 Reporter 假装搜索、执行代码或重新打开附件 |

## 两个完整走读

### “查今晚某影院场次”，地图访问失败

Scheduler 先确定城市、影院分店、日期和必须返回的字段，再安排 Web。web-cinema-showtimes 指导 Worker 核对影院、场次与票价；只有另有起终点路线需求才考虑 web-map-routes。公开票务页只是候选，不保证能访问。

工具返回高德 403 后，换影院/运营者公开页或票务场次页，而不是无变化地反复请求。若只有地址有证据、场次仍缺，提交真实引用和缺口。Reporter 可选择 report-web-evidence + report-execution-failure：确认地址的证据范围，场次不满足；记录“工具返回 403”，不扩大成“高德永久不可用”或猜测反爬原因。

报告写清哪项标准未满足、已获得什么、下一次需要什么不同路径。Scheduler 只在有新来源/条件时安排补证；没有新路径则报告未核实并停止。另一来源已覆盖所有 MUST 时，某个地图失败不应使整步失败。

### “生成 CSV 筛选工具”，Reviewer 看不到 Worker 对话

Scheduler 使用现有 code_task：requirements 区分 MUST/SHOULD；interfaces 的真实 kind 可为 CLI/PYTHON_SYMBOL/FILE_ARTIFACT，details 写明参数、格式、字段及错误规则；validation_expectations 要求小样本筛选与执行后重读；non_goals 限制不做无关功能。Step success_criteria 与这些内容一致。

Worker 主要选择 code-transform-files；CLI 本身也是验收点才加 code-build-python-cli。Reviewer 独立选 review-file-output，从契约构造匹配、不匹配、非法记录，按真实入口运行并重读。

测试收集失败先核对路径/解释器，Reviewer 误用参数先修测试；候选遗漏合法行才给 Worker 具体返修，包含 requirement ID、最小输入、预期/实际与复跑方式。新候选重测失败用例和原先正确路径。文件存在、测试零收集或 Worker 自述通过，都不是独立完成证明。

## 真实能力与不做的事情

- 本轮实际启动隔离的 Code Worker/Reviewer 容器验证，读取到 Python3.12.14、pytest、Node，未发现Pydantic、Playwright、Chromium；网络none。HTML视觉/交互仍未验证。完整共享路径、收集失败与返修重测见本轮记录。
- 邮件当前源码有 6 个读取/下载别名和新的 email_create_draft。Drafts 工具仅纯文本 APPEND，需要地址与主题/正文，无附件/回复链参数；没有发送能力。本轮只核对声明和离线测试，没有连接邮箱或操作邮件。
- 草稿工具存在时才有 general-save-email-draft 候选。仅“帮我拟回复”输出对话文本；明确“保存 Drafts”才写入。DRAFT_SAVED 与 sent=false 是草稿回执，超时可能已写入，不能盲重试。
- 提醒有创建、查询、暂停、恢复、软删除、历史和即时通知；没有 schedule_update，也没有月度/工作日重复。服务要运行，SUBMITTED 只是 Windows 接受通知。本轮没有创建提醒或弹通知。
- 飞书文件回传仍是原有独立审批能力；没有新增发送 Skill 或实际发送。邮件限制不能通过飞书、浏览器、脚本等渠道绕过。
- 工具存在于源码，不等于当前账号已配置或当前角色拿到了该工具；运行时的能力与权限过滤仍是准入依据，Skill 不是安全边界的替代物。

## 参考与选择理由

前轮公开 Skill/Agent 模板和取舍见 [策略参考](skill-policies.md#公开参考与取舍)。保留“触发条件—行动—检查—失败—停止”的结构，按本地角色独立改写；不安装外部 Skill，也不照搬未经适配的浏览器、联网或无限重试要求。

- [pytest 官方执行指南](https://docs.pytest.org/en/stable/how-to/usage.html)：实际收集、目标用例与执行入口；用来指导先排查验证环境。
- [JSON Schema 官方对象约束](https://json-schema.org/understanding-json-schema/reference/object)：必填、额外字段、缺失与 null 的差异；用来设计有区分力的正反例。
- [MDN 测试策略](https://developer.mozilla.org/en-US/docs/Learn_web_development/Extensions/Testing/Testing_strategies)：按功能及平台分层检查；改写为当前沙箱能做/不能做的验证证据。
- [猫眼移动影院目录](https://i.maoyan.com/cinemas)：中文影院实体/地址/起价查询样例；搜索可索引不保证正文可访问或实时场次可查。前轮实际读取抽样见 [Web 记录](worklogs/2026-09-06-web-source-policy.md)，不维护“永不 block 网站榜”。

## 验证与 token 预算

前轮34项/68项离线回归属于历史记录。本轮38份格式校验通过，测试中的名称和新增Web单项/组合隔离用例已同步；最新回归数量见 [本轮验证](skill-role-field-test.md)。选择模型仍是脚本返回，不能用接线通过声称实际模型选择准确率或遵守程度。

cl100k_base 的这组历史估算中，正文合计22,377 token，但全库不会注入一个角色；单项402–1,324。Web链路现已取消独立Web Reporter与第二次报告轮次，Web Worker通过一次 `complete_web_step` 提交事实，Harness确定性生成StepReport；因此旧的“Web Reporter + 失败审核”代表性加载量不再是当前运行结构。候选描述、原任务、协议及证据另计，不是服务商实际账单或缓存收益。

未运行付费模型、AppWorld、SkillOpt 或无人值守批次；未验证真实邮箱、桌面通知和浏览器行为。本轮Docker演练由主代理手动替代角色，不是模型自主端到端成绩。
