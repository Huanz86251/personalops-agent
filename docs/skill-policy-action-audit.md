# 38项Skill逐句检查与真实行动闭环

2026-09-06。逐份检查name、description、roles、topics、required-tools及全部正文；判断每句是否改变下一步、是否要求不存在的工具、是否把局部/历史/参考结果冒充最终结果。38份正文均有针对性修改，保留原有角色分域，不声称每个业务都已联网实跑。

## 真实换路结果

| 障碍 | 改变方法 | 实际结果与边界 |
| --- | --- | --- |
| 淘宝限定查询无结果；列表文本空，浏览器验证码 | 精确型号转ZOL/值得买，用browser_find定位 | M650-M参考价269元、京东链接；194元优惠正文为2023-01-26旧价。不是淘宝指定店现价 |
| 高德地点获取失败并出现登录，百度提交路线遇安全验证 | 停止受阻路径，另用公开地铁服务验证独立样例 | ExploreShanghai选虹桥火车站→路线查询→上海科技馆，返回2号线15站、估算47分钟/5元。不是莱西答案、门到门时间或官方保证 |
| 中国电影博物馆影讯文本乱码 | 浏览器读同页嵌入内容，定位日期组及场次行 | 9月6日至10日分组；9月6日1号厅《八仙！》09:50–12:14、43元起。不证明余座/结算价 |
| 陶陶居总店混分店，探店正文曾403 | 街道消歧→品牌具名门店目录 | 取得第十甫路总店地址/电话，不替代今日营业时间 |
| arXiv摘要/HTML连接重置 | 精确题名→会议论文集 | NeurIPS作者、会议、摘要与Paper链接，不冒充已读PDF实验 |
| 商品直链只得到京东壳 | 功能问题转厂商同地区页面，按next_index继续 | 罗技M650的SmartWheel及连接说明，不证明库存或店价 |

公开入口：[ZOL](https://detail.zol.com.cn/1970/1969107/param.shtml)、[值得买旧优惠](https://www.smzdm.com/p/71385750/)、[ExploreShanghai](https://exploreshanghai.com/ditie/)、[电影博物馆影讯](http://www.cnfm.org.cn/cgzn/jqyx.shtml)、[陶陶居目录](https://www.taotaoju1880.com/zoujintaotaoju/taotaojumendian)、[NeurIPS](https://papers.nips.cc/paper/7181-attention-is-all-you-need)、[罗技M650](https://www.logitech.com/zh-cn/shop/p/m650-signature-wireless-mouse)。

这些是本机当次可用路径，不是永久网站排名。没有绕验证码、登录或使用未知镜像。莱西月湖公园→横店影城是选定的公开地点样例，未使用用户定位，也未取得可靠距离；上海地铁为另一个独立样例，不算完成莱西请求。ExploreShanghai的方向文案不清晰，不能猜终点方向。影院“43元起”没有选座或结算证据。

## 每份Skill的审查结论

元数据：名称/目录一致，无旧别名；描述区分任务而非穷举工具；roles对应实际角色，topics只辅助发现；required-tools与真实注册核对。MCP参数以本轮返回Schema为准（如target），不把本次元素ID写进可复用正文。

| Skill | 逐句审查后的决策要点 | 依据 |
| --- | --- | --- |
| web-shop-prices | 型号转比价/优惠社区，保持卖家与价格口径 | 本轮真实取证 |
| web-map-routes | 按交通方式换服务，区分站到站与门到门 | 本轮真实MCP |
| web-cinema-showtimes | 乱码换浏览器，日期组与具体行配对 | 本轮真实MCP |
| web-local-services | 街道消歧、具名目录补身份 | 本轮真实取证 |
| web-venue-visits | 保留减词成功例，解码障碍用浏览器 | 前轮官网+本轮MCP |
| web-product-plans | 缩减失败清单，规格/参考价分来源 | 本轮真实读取 |
| web-technical-docs | 保留同版本中文检索例，资料不冒充运行证明 | 前轮真实取证 |
| web-official-records | 错误页→准确标题→可追溯转载 | 前轮真实取证 |
| web-research-papers | arXiv失败→会议页成功，限制摘要结论范围 | 本轮真实取证 |
| web-source-investigation | 字数不硬判定，长正文取可交接短证据 | 本轮读取+回放 |
| code-implement-core | 正常样本后补能区分等值错误的输入 | 前轮真实Docker |
| code-debug-root-cause | 边界差异定位比较符，不一起改解析/环境 | 前轮故障注入Docker |
| code-build-python-cli | 实测stdout/退出码/原输入状态分别检查 | 前轮真实Docker |
| code-transform-files | 中文/分隔符/空格路径和拒绝覆盖后回读 | 前轮真实Docker |
| code-validate-schema | 标准库入口验证0、缺失/null/bool等 | 前轮真实Docker |
| code-build-web-ui | 无浏览器时逻辑与DOM验收分流 | 能力实测，UI未跑 |
| review-code-behavior | 契约推导预期，不复制Worker样本 | 前轮真实Docker |
| review-python-cli | 收集失败修入口，断言失败定位产品 | 前轮真实Docker |
| review-file-output | 输出改到/review，不放宽/workspace只读 | 前轮真实Docker |
| review-schema-contract | 不为标准库JSON测试强加Pydantic | 前轮真实Docker |
| review-web-ui | 具体说明键盘/布局缺口，不重写已测逻辑 | 能力分流，UI未跑 |
| review-code-state | 重建运行时核对候选与恢复角色 | 离线恢复回归 |
| plan-code-acceptance | MUST写公开行为，不给测试答案 | 前轮CSV契约 |
| plan-task-dependencies | 真实可读交接才构成依赖，/tmp不共享 | 前轮Docker |
| plan-uncertainty | 先查改变方案的依赖，再按结果分流 | 容器预检 |
| plan-web-research | 参考行情/指定店现价分验收 | 本轮取证与回放 |
| plan-recovery | 零收集与两条断言失败派给不同责任方 | 前轮Docker |
| plan-personal-tasks | 保存未知先核对，不扩展发送/重复创建 | 接口、离线替身 |
| report-web-evidence | 相同参考价对不同契约得不同结论 | 真实证据手工回放 |
| report-execution-failure | 下一步含变化变量与成功信号 | 真实失败/恢复 |
| report-artifact-delivery | 内容/发布缺口分流，不重做产物 | 导出及发布回归 |
| synthesize-evidence | 换源获得数字不降低原标准 | 本轮回放 |
| general-read-documents | 预览截断但全文已保存→回读，不重复转换 | 真实本地转换测试 |
| general-compare-options | 历史/参考价不直接排名，未知单列 | 本轮价格材料 |
| general-read-email | 不猜筛选参数，近期未找到不代表全邮箱没有 | 声明/白名单替身 |
| general-save-email-draft | 区分拟稿/保存；超时保持未知不盲重试 | APPEND替身，未写邮箱 |
| general-manage-reminders | 修时区字段；历史用ALL防漏项 | 临时SQLite/虚拟通知器 |
| appworld-execute-api | 示例明确未实跑，按ID回读再局部重试 | 适配回归，未跑任务 |

## Reporter可见性与17个字段

真实ZOL后段12000字符进入StepReviewPacket后，6000字符头尾裁切丢了“参考报价”标签；别处可能仍有价格数字，反而容易丢失价格口径。真实browser_find短结果保留型号和标签。所以改Worker提交策略：引用目标区间/定位工具结果，不依赖自己的转述，不无限扩大上下文。

| 字段 | 应写什么及决策用途 |
| --- | --- |
| step_id | 当前契约ID，防串步 |
| status | 每项MET才COMPLETED；部分成果、缺条件、执行失败分开 |
| summary | 达成范围，不重放轨迹 |
| stop_reason | 本次为什么结束，不冒充错误根因 |
| criterion_results | criterion原样有序，status反映证据，evidence带调用/定位 |
| confirmed_results | 有证据且保留口径的事实，供复用 |
| assessment_source | 独立Reporter用INDEPENDENT_REVIEW，不能伪装执行者 |
| completed_work | 已做动作，与目标达成分开 |
| artifacts | Reporter留空，宿主发布后填真实path/description |
| approved_artifact_refs | 仅选已验证的resolved_artifacts.review_ref |
| worker_contributions | 实际worker_id与有依据的contribution，无身份不编造 |
| evidence | 最小必要索引，不复制网页 |
| errors | 调用/路径→观测→影响；已恢复错误可保留 |
| unresolved_items | 当前剩余字段/能力，不抹掉已验证成果 |
| next_action | 保留什么→补哪项→换什么→怎么验收→停止/输入条件；完成可空 |
| request_replan | 原计划需要改变才true，不等于有任意错误 |
| replan_reason | 配合true说明哪项原前提失效 |

发现并修复真实校验缺口：UNKNOWN/PARTIAL/NOT_MET标准与整体COMPLETED的矛盾报告原来可通过。先写5项回归得到4个失败（其中一个方法有3个子案例），修复后通过；矛盾报告进入现有有界修复/安全兜底。保留“早先错误已恢复、最终全MET”的正常完成，不用errors非空一律失败。

手工扮演Reporter：相同真实材料，对“参考价”给COMPLETED，对“指定店现价”保留UNKNOWN/PARTIAL。真实Scheduler事实写入和压缩保留全部17字段，不含Worker对话。报告文件明确标manual；这证明可见性和接线，不证明模型自主决策质量。未新增报告Schema字段。

## 证据、测试与清理

- 新16次web_search、16次fetch_webpage，原工具记录从 `.agent/web-source-probes/20260906T053232_6625e3f2.json` 起独立保存。
- 真实MCP会话 `.agent/browser-skill-probes/20260906T053441_e65b9f27/`：schemas.json和001–021结果。独立headless profile、已缓存0.0.78包、npm离线，无登录态；先前一次无PTY启动因stdin EOF退出，仅发现Schema。有效会话已正常关闭，快照与证据保留。
- 真实证据回放 `.agent/skill-policy-handoffs/20260906T054912_a77573e0/`，包含审核包、manual报告、Scheduler事实。
- 38份格式验证通过；三组离线回归40、45、51项通过（有重叠，不合并成不重复总数）。真实转换/临时存储与外部替身如上表。现存LangGraph类型提醒、nul ResourceWarning仍在。
- 仅删除skills下34个已确认空的旧命名目录，逐个验证绝对路径和仍为空，没有递归删除文件。目录可重建，里面没有内容需恢复；当前skills空目录0。
- 未调用付费模型、AppWorld任务、真实邮箱/草稿/提醒/发送；未改邮件实现、Docker镜像、权限、业务预算或登录配置；未commit/push。

复跑入口：`scripts/probe_browser_skill_paths.py`显式交互公开页探测；`scripts/probe_skill_report_handoff.py`只回放传入观测；`tests/test_skill_policy_handoff.py`检查状态一致性、有限证据和字段传递。动态价格/场次/可达性需在未来任务重验，不把本轮缓存当实时答案。

## 后续再验证入口

2026-09-06继续通过真实项目工具补到了苏宁商家标价、Apple官方商店报价、同一家莱西影院的场次，并真实下载邮件附件后复现生产路径不可读。以[业务字段再验证](skill-live-business-recheck.md)和现有Skill正文为最新结论；本文前序的未验证项是当时状态，不再代表所有入口均失败。
