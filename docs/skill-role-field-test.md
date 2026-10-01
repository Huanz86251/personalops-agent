# Skill 命名重规划与 Code 角色真实演练

2026-09-06。用户明确当前未上线，不需要旧名称兼容，并要求真正验证Docker、角色上下文隔离、文件交接和独立测试，再调整方法。

## 当前名称设计

以“角色/用途—任务场景”命名，38项完整目录见 [skill-scenarios](skill-scenarios.md)。Web10项、Code6项、Reviewer6项、Scheduler6项、Reporter3项、General5项、共享综合1项、AppWorld1项。

- `web-*`：取证场景，不按每个网站造一个Skill。
- `code-*`：实现/调试方法；`review-*`：独立验收方法。
- `plan-*`：规划/恢复决策；`report-*`：结果与失败归因。
- `general-*`：实际具备工具的个人事务/资料处理。
- `synthesize-evidence`：跨角色证据综合；`appworld-execute-api`：仅模拟环境方法。

旧34个 SKILL.md 已删除或迁移到新路径，没有旧ID别名和兼容映射。选择器、执行中的正文交叉引用、离线测试、当前目录说明同步替换。历史日志/旧运行证据不改写；旧快照的正文仍是原有快照机制保存的历史事实，不等于目录有兼容别名。空旧目录不参与发现；清理空目录命令被执行层拒绝，未尝试绕过，实际旧SKILL文件已移除。

### Web 拆分边界

| 用户任务 | 方法 | 不应顺带加载 |
| --- | --- | --- |
| 淘宝某店某规格报价/跨店比较 | web-shop-prices | 厂商订阅、影院票价规则 |
| 博物馆开放、入口、参观条件 | web-venue-visits | 影院排片；仅问地址不自动加地图 |
| 影院排片、场次、余座、票价 | web-cinema-showtimes | 普通场馆参观规则 |
| 从起点到终点、耗时、方向/末班 | web-map-routes | 不把附近车站当作完整行程 |
| 餐馆/门店/生活活动 | web-local-services | 上述四类的完整规则 |
| 厂商规格、订阅与通信套餐 | web-product-plans | 没有SKU报价需求时不加网店方法 |

每项独立可用；需要两种不同成果时才组合，0–3上限不变。含引导的cl100k_base估算：本地服务639、场馆717、影院695、路线690、店价799。无需为了保持统一而加载旧的全生活大策略。

### 淘宝本轮小探测

沿用项目真实 `web_search_tool`，无登录或模型调用。`site:item.taobao.com 罗技 M650 鼠标 价格` 与 `site:detail.tmall.com 罗技 M650 鼠标 官方旗舰店` 两次均 No results；撤域名后 `罗技 M650 淘宝 价格` 遇搜索后端 RequestError/DecodeError。记录在 `.agent/web-source-probes/20260906T050100_f9b4b477.json`、`20260906T050102_37b43d5c.json`、`20260906T050118_fb683825.json`。

没有取得商品正文，故不推断淘宝一定要登录、商品不存在或任何价格已核实。网店方法按可观察SKU、卖家、优惠资格和运费设计，不声称匿名抓价稳定可用；不加购、领券、下单、支付或联系商家。

## Code 实测装置

使用当前 `CodeSandboxManager` 与 `DockerSandboxBackend` 的真实 create_pair、upload/download、execute、handoff、freeze、export_candidate/export_review。脚本为 `scripts/probe_code_skill_handoff.py`。复用已存在镜像，明确 `auto_build=False`；预检只读，不允许脚本启动Docker服务。没有调用模型/Agent工厂、外层Planning图或Publisher审批工具，因此这是底层真实交接演练，不是整个产品端到端通过。

实测前Docker服务版本29.6.1；镜像 `personalops-code-sandbox:py312-v1`，ID `sha256:aee974848c2dbacfe1ee2176b6fe9c4ccadf7f6e8bcb9afd0bc00b14dfaf934e`。两角色非root、network=none、只读根文件系统、cap_drop ALL；本次容器内Python3.12.14，pytest和Node存在，Pydantic/Playwright/Chromium未发现。未下载依赖或修改镜像。

### 微项目与独立预期

小项目读取中文CSV，用JSON配置筛选并输出CSV：金额大于等于阈值（零合法），可选类别，保留顺序与文本；拒绝非法配置和金额、不覆盖已有文件；成功stdout为JSON。

先按公开契约编写Reviewer检查，再编写候选。契约与用例源文件分别是 `tests/fixtures/code_skill_probe/contract.json`、`review_checks.py`，实现为同目录 `filter_report.py`。脚本将一处 `>=` 明确变成 `>` 生成故障候选，修复后再传干净实现；这是**人为注入的演示缺陷**，不是声称自然发现了产品代码缺陷。

Reviewer执行环境只有公开契约、候选、输入与自己的用例，没有Worker对话、私有临时状态或自测脚本。主代理手动替代两个角色，主代理自身知道演练设计；这证明无Worker对话的工具环境可以独立执行契约测试，不能证明两个模型在认知上盲测或会自主生成同样测试。

### 文件权限与交接

| 路径/材料 | Worker | Reviewer | 实际证据 |
| --- | --- | --- | --- |
| `/handoff`输入与契约 | 可读、不可写 | 可读、只读挂载 | Worker读中文CSV成功，写入失败；Reviewer读契约成功，inspect确认两侧只读 |
| `/workspace`候选与Worker输出 | 可写 | 可读、不可写 | Reviewer读到相同源码字节和真实CSV，覆盖候选返回只读错误，前后hash未变 |
| `/review`测试与JUnit结果 | 无Reviewer卷 | 可写 | Reviewer创建/运行测试；交回Worker后看不到测试文件 |
| Worker `/tmp`私有标记 | 本容器临时状态 | 不可见 | Reviewer检查该路径不存在 |
| 候选/测试导出到宿主私有目录 | 宿主控制 | 宿主控制 | 真正调用两个export方法，版本与测试文件保留；不冒充Publisher已正式发布 |

两角色顺序交接，不是同时运行同一可写候选。容器路径不等于宿主路径；Worker在/tmp写东西不能当作交付。程序控制的导出目录才是宿主可见证据，不自行把 `/workspace/foo.csv` 当用户电脑文件链接。

## 真实失败、修正、复验

1. Worker正常样本运行成功，stdout为 `{"written":1}`，生成可重读中文CSV。这不足以发现边界缺陷。
2. Reviewer默认目录为 `/review`。直接 `python -m pytest /review/test_contract.py --collect-only` 因找不到 `/workspace` 内模块而失败，退出2、零收集。此时不能判定候选业务失败。
3. 先确认源码实际存在，再按平铺布局设置 `PYTHONPATH=/workspace`；同一用例收集到12项，退出0。没有降低断言或改产品。
4. 故障候选运行12项：**10通过、2失败**，分别为零阈值与等于10阈值，精确暴露 `>` 违反 `>=`。
5. 交回Worker修复，再回Reviewer；源码字节确已变化，原用例不变。复验 **12通过**。
6. freeze后导出两版候选与Reviewer测试/JUnit，核对版本哈希不同；再清理本次拥有的两个容器、两个数据卷，逐个inspect确认不存在。无全局prune、无宿主数据删除，历史证据保留。

测试覆盖正常筛选、类别/顺序、零和等值边界、缺失/null/bool/字符串/额外配置字段、非法金额与原输出保留、禁止覆盖、中文逗号换行/空格路径、空输入文件头。不是格式/异常/性能/安全的穷举，也未验证浏览器或Pydantic项目。

## 根据实测修改的方法

- 固定Code/Reviewer协议说明 `/workspace`、`/handoff`、`/review` 的权限与交接；这属于执行前必须知道的边界，不依赖是否选中Skill。
- `review-python-cli` 加入实际import/collect诊断与平铺目录示例，区分收集失败与断言失败；其他包布局不能机械照抄。
- `code-build-python-cli` 删除“不得修改环境变量”的过宽禁令，允许明确可复现的入口路径配置，禁止通过关闭检查掩盖错误；交付必须说明Reviewer可复跑条件。
- `code-transform-files` 修正此前含糊的 `/artifacts` 假设，明确交付在 `/workspace`，不用 `/tmp` 作交接。
- `review-file-output` 不照抄Worker写 `/workspace` 的命令；独立测试输出指定 `/review`，只读错误不判产品缺陷。
- `code-validate-schema` / `review-schema-contract` 不假定Pydantic存在，按真实入口检验缺失/null/bool/零等语义，不为测试强加依赖。
- `plan-code-acceptance` 明确公开CLI、输入/输出位置、依赖和错误约定，避免Reviewer必须猜Worker上下文。

没有修改Docker隔离、模型选择、执行预算或权限。没有新增“自动安装环境”或“跳过审核”的策略。

## 留存证据与验证边界

真实运行目录：`.agent/code-skill-probes/20260906T050922_f7ef8f9b/`，含 result.json、handoff、candidate-r1、candidate-r2 和 review（两次JUnit、测试及临时样本）。旧结果不改写。独立用例模板后来仅从 test_contract.py 改名为 review_checks.py，避免仓库测试发现器把容器夹具当本机测试；上传到容器时仍叫test_contract.py。

38份Skill格式校验通过。路由/快照/Reporter/Code/发布/沙箱/运行时/AppWorld适配/提示上下文共86项相关离线回归全部通过（208.982秒），不与容器内12项契约检查合并计数。离线回归使用测试替身，脚本模型的选择结果不计为模型质量；另有上文真实Docker证据。测试过程仍有现存的LangGraph自定义类型反序列化提醒及nul文件ResourceWarning，不影响本次退出0，但不表述为零警告。未执行付费模型、AppWorld任务、实际邮箱/发送/提醒或浏览器操作，未commit/push。
