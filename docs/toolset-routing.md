# Cross-Encoder 工具组路由

## 当前结论

工具组路由不再要求 Qwen 生成标签或 JSON。Conversation Runtime 启动时已经加载的 BCE Cross-Encoder 被同一进程复用；在 `MEMORY_MODEL_DEVICE=auto` 且 CUDA 可用时，它常驻 GPU，直到 Runtime 关闭才统一释放。路由不会创建第二份模型实例。

当前主流程分两层执行：Scheduler 在规划前缩小能力目录；每个真正持有业务工具的 General、Web、Code Worker 与 Code Reviewer 在自己的第一轮模型调用前，用同一个 Step 查询和同一个 Cross-Encoder 再路由一次。纯 Step Reporter 和 Final Reviewer 没有业务工具，不运行这一步。

1. 按该角色的真实工具池过滤掉缺少 required tools 的组。
2. Worker 优先使用 Scheduler 的 `rag_query`；没有时使用当前 Step 的目标与验收条件组成稳定短查询。
3. 角色控制工具、报告工具、执行历史以及 RAG 工具先作为基座候选移出业务竞争。
4. 将查询与所有可用业务组的正向能力文本组成 pair，一次 batch 交给 Cross-Encoder。
5. 选择最高分主组；非独占主组可合并近分副组，最多三个。独占主组只保留自己的业务工具。
6. 应用确定性的覆盖关系，避免宽组和被它完整包含的窄组重复出现。
7. 将角色基座候选确定性加回并去重，再依次执行每种工具自己的最终可用性检查。
8. `NO_TOOL` 必须同时满足绝对阈值，并明显高于全部业务组。模型异常、无主组过阈值或结果不确定时，保守回退为该角色原有完整工具池。

RAG 是第一种带最终阀门的基座工具：只有 Scheduler 明确填写非空 `rag_query`、Harness 实际召回形成当前执行者的 grant 时，`search_knowledge` 和 `read_knowledge` 才会留在最终工具列表。缺 query、零命中、召回失败或 grant 属于别的 Worker 时，两项都会在模型调用前删除。以后新增带条件的公共工具，应沿用“基座候选 → 工具自己的最终 gate”这一层，而不是把权限判断塞进 Cross-Encoder。

真正的工具对象、参数 JSON Schema、角色白名单、审批和权限仍由原运行时强制执行；相关性模型只缩小候选，不能授权工具。

## 能力卡和硬编码边界

每个 `ToolsetSpec` 硬编码以下内容：

- 稳定组名；
- 提供给 Scheduler 的短描述；
- Cross-Encoder 能力卡；
- 主选择阈值；
- required / optional 工具名称；
- 选中后交给执行模型的工作说明。

真实函数参数 Schema 继续来自 LangChain Tool，不复制到能力卡。把完整参数 JSON 塞给相关性模型会增加噪声；路由阶段只需判断能力域，执行阶段才需要精确参数 Schema。
`scripts/tool_inventory.py` 会同时输出每组的 required/optional 工具名和每个真实工具的完整执行 Schema，可据名称确定性关联，避免在两处复制后发生漂移。

能力卡位于 `prompts/routing/toolsets/`，统一包含：选择它、不选择它、典型命令、易混淆边界。完整文件供维护者审查；实际 Cross-Encoder 输入只抽取“选择它”和“典型命令”。原因是通用 Reranker 可能把反例中的关键词也当成相关证据，不能假设它可靠理解 Markdown 中的否定关系。

当前真实组：

| 组 | 激活条件和边界 |
| --- | --- |
| FEISHU_FILE_EXPORT | 仅真实工具池存在飞书回传工具；仍需宿主确认 |
| LOCAL_DOCUMENTS | PDF/PPTX/OCR/转换/Excel |
| LOCAL_ANALYSIS | SymPy、Python 语法和 Ruff 静态检查，不执行代码 |
| WEB_RESEARCH | 搜索、读取、核对公开信息，不执行网页动作 |
| BROWSER_AUTOMATION | 页面导航、点击、输入、表单和网页邮箱 |
| FILE_INSPECTION | 只读查找、目录、文本和源码检查 |
| FILE_EDITING | 创建和修改文本文件 |
| SOFTWARE_DEVELOPMENT | 运行、测试、构建和完整开发验证 |
| APPWORLD | 独占业务组；General/Code Worker展开 discover+execute，Code Reviewer展开 discover+verify |

另有一张 `NO_TOOL` 卡，它不是业务工具组。

## 分数规则

当前 BCE 分数经过 sigmoid，但不把它描述为经过业务校准的概率。

- 大多数业务组主阈值：0.35。
- 飞书回传主阈值：0.40，降低“写入/交付文件”产生的误触发。
- 次级候选允许比自己的主阈值低 0.05，但必须与最高业务分相差不超过 0.10。
- `NO_TOOL`：至少 0.45，并且比最高业务分至少高 0.05。
- 最多三个业务组。

这些值是首轮本机 smoke 样例的工程初值，不是充分数据校准后的最终参数。正式校准应为每组准备独立 Train/Dev/Test 命令，重点加入易混淆 hard negatives，冻结 Test 后再调阈值。

## 当前覆盖关系

- FILE_EDITING 覆盖 FILE_INSPECTION。
- SOFTWARE_DEVELOPMENT 覆盖 FILE_INSPECTION 和 FILE_EDITING。
- APPWORLD 是独占业务组；它为最高分主组时不再附带任何其他业务组。角色报告、历史和满足最终 gate 的 RAG 基座仍可保留。

覆盖只用于结果去重，不会改变工具权限。

## 评测和可观测性

`toolset_routing` span 保存候选组、阈值、全部分数、最终组和是否回退；物理模型调用仍由 `local_cross_encoder.predict` span 记录。`ToolsetRouteDecision.route_scores` 也保留本次分数，便于离线评测。

可运行 `scripts/toolset_router_smoke.py` 做不调用云模型的本地检查。当前 11 个手写覆盖样例在 BCE/CUDA 上预期组召回 11/11；这个小样本只能证明接线和初始边界可工作，不能声称生产准确率。

后续正式比较至少记录：每组 Recall、Macro/Micro F1、无工具拒识、过路由率、完整目录回退率、P50/P95 延迟、最终任务成功率。只有真实路由样本积累后，才考虑微调判别模型；不需要为了这条链路调生成模型。

## Qwen 边界

工具组路由已不再读取 `prompts/routing/toolset_router.md`，也不调用 `classify_many_with_router`；旧生成式工具路由提示词已删除。

项目中的 Qwen 名称没有被全局删除，因为它还可能属于独立的 Memory Router 配置和 Qwen3Guard 注入检测链路。它们不是工具组路由，是否删除必须分别验证，不能借本次改造误删。
