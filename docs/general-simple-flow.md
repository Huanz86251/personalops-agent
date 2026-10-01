# General 简单执行流程

2026-09-05。按用户本阶段指令，General 与 Web/Code 的验收流程分开。

## 执行与汇报

```text
Scheduler 选择 GENERAL
  → 轻量 LangChain Agent（业务工具＋受限文件工具）
  → 工具执行；General 自行检查结果
  → report_general_result
  → general_report 节点直接整理 StepReport，无额外模型调用
  → 后续步骤，或由主模型汇总回答
```

General 不再使用 Deep Agents 构造器，不提供默认的 `task` 子任务工具、
`publish_worker_progress` 或 Web 专用的 `complete_web_step`，不进入独立 Step Reporter。
未增加独立 summary 模型调用，也不会为整理输出再强制调用一次模型。
Web 保留进度/提交/独立 Reporter，Code 保留自己的 Reviewer 和控制流程。

General 保留 LangGraph checkpoint、事件 pause point、文件访问边界和调用预算。
`WorkerLeadershipBridge` 在 General 仅承担恢复/事件控制，不启用模型进度评估。
General 的失败/阻塞不会触发通用自动重试，也不自动更换 Worker 反复验收。
任务级主模型仍可根据报告决定后续计划或最后汇总；这不是 General 的独立验收。

完成工具的输入是小型结构：`status`、`summary`、`unresolved_items`、
`evidence_tool_call_ids`、`files`。文件只需要已有 `/artifacts/` 路径、说明和可选
输出槽 `output_id`，不要求 General 填写代码验收或远程仓库字段。runtime 参数
不暴露给模型。工具依据真实 ToolMessage 和文件状态检查引用，不调用另一个模型。

报告标明 `assessment_source=GENERAL_SELF_REPORT`。`COMPLETED` 是 General 自报，
不会被写成独立验证通过的 `confirmed_results`；有未解决项时降为 PARTIAL。
自然语言直接退出不会多花一轮模型补格式，运行时保留文字并标记 PARTIAL。
没有结果或运行错误按失败处理。旧 checkpoint 中只有 submission 的文件仍可沿用
已有文件校验/共享机制，但不会据此推断工作完成。

## 文件和上下文

上传原件仍位于本次 run 的只读 `/handoff/inbound/...`，General、Web、Code 的
文件访问路径没有拆散。General 生成的文件在 `report_general_result.files` 中
列出后，由现有 Publisher 做存在性、路径、输出槽和内容校验，再复制到共享
handoff；主模型和下一 Worker 获得实际共享路径。这个 Publisher 不做内容模型审核。
已有用户文件交付/工作区合并规则不因 General 自报而被删除。

原始用户指令和当前 Step 一次放入 Worker 初始消息，多轮工具调用继续使用已有
历史，不反复追加原指令或附件清单。系统提示和工具定义保持稳定；较大的 PDF、
提取结果和中间文件按需读取。恢复已有 checkpoint 时，继续使用存储状态而不是
再次追加起始输入。新一轮任务会清除上一轮 General 完成标记。

“不重复追加”不等于“后续 API 请求不带历史”。常规模型请求仍包含当前需要的
历史上下文。本阶段验证了请求前缀稳定和原指令不重复增长，未调用 DeepSeek
实测缓存命中率或费用，也没有引入不断膨胀的统一任务输入字段。更完整的资源
目录/按需加载协议留待后续阶段。

并行开发中的 Skills 功能后来接入了同一入口，本阶段保留了该功能，并修正其
SkillsMiddleware 构造参数与当前安装 SDK 的不兼容。配置为动态选择技能时，
入口可能另有选择模型调用；它不属于 General 的独立验收，也不能把它计为已经
省掉的费用。固定/关闭动态选择的配置由 Skills 功能单独管理。

## 用户案例复核

| 请求 | 当前结论 |
| --- | --- |
| 飞书发送 PDF，要求读取/总结 | General 可读取 handoff 原件，自动 OCR，直接自报 |
| 根据 PDF 搜索资料 | 建议 General 先读取，再由 Web 搜索；路径和摘要可交接；真实模型的自主路由尚未在线实测 |
| 在指定磁盘检查有没有同一 PDF | 尚不能完成：没有提供给 General 的主机磁盘检索和流式文件哈希比较工具；附件保存时的 SHA-256 校验不等于全盘查重 |
| 把本地 PPT 发送到飞书 | 尚不能完成：需要本地文件选择/登记及飞书文件上传发送工具；当前入口仅接收附件 |
| 检查电脑电源/运行状态 | 只能依据实际可用的状态工具判断，目前没有新增系统状态工具；断电的主机自身无法执行查询 |

后两类文件操作适合未来作为 General 的本地工具接入，不需要为了这些操作启动
Code Reviewer。但这不是本阶段已经实现的能力，不能用模型猜测或把上传 key 当路径。

## 验证

`tests/test_general_simple_flow.py` 覆盖真实 General 工具调用、Scheduler 到
general_report 路由、无独立 Step Reporter、无 General 重试、产物进入 handoff、
预算耗尽不追加模型轮次、同一线程新任务可运行、稳定请求前缀和只读输入边界。

相关回归覆盖飞书真实样例 PDF/OCR、General、Web/Code 规划、Web 进度/提交、
事件暂停和恢复。原来借用 General 工厂测试“提交验收”的测试已改用 Web 工厂，
继续验证 Web 的原有协议；没有让 General 假装保留该协议。

验证结果：主要回归 80 项通过（77.18 秒）；本地原生工具另外 14 项通过；补充只读
边界测试后 General 专项 9 项全部通过（14.71 秒，其中 8 项与主要回归重复）。
此时合计覆盖 95 个不同测试。随后补充“无预算未启动也不进 Reporter”测试，
以及共享 Skills 改动兼容修复后，General 10 项与 Code 规划 4 项再次全部通过
（17.29 秒）。主要回归只有 Feishu SDK 的两条既有弃用警告。

最终对包含并行 Skills 改动的工作区重新验证：General、飞书图片/PDF、Web 控制
协议和 Code 规划等 **54 项通过（52.28 秒）**。核对了关键接入源码哈希，确认这次
验收期间所检查的 General/Planning/Agent 接入文件保持一致。

没有重启当前 Agent，没有真实 Feishu 发送或付费模型调用。新流程在下次正常启动
后加载，在线行为和费用仍需后续实际验收。
