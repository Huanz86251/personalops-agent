# Web Search: Google 主搜索、百炼回退与微信公众号补充

项目的 `web_search` 工具默认通过 Bright Data SERP API 查询 Google；主后端连接、鉴权、限流、返回格式异常或没有有效网页结果时，自动回退到百炼 WebSearch MCP。工具名保持 `web_search`，结果继续使用 `title`、`href`、`body` 字段，供现有 Web Worker 使用。

当模型显式传 `search_service="wechat"` 时，同一工具改用 Search1API 的 WeChat 搜索，不经过 Google/百炼。微信模式每组固定 `max_results=3`、`crawl_results=3`：最多返回3条文章，并尽量让3条都附正文；正文在进入模型前每条最多保留8000字符，避免一次搜索把上下文挤满。主query和最多2个真正互补的 `alternative_queries` 通过现有批量入口并发执行，进程级并发上限仍为3。

## 配置

1. 在 Bright Data 创建 SERP API zone，并准备对应 API Key。
2. 在百炼控制台开通 WebSearch MCP，并准备可访问该服务的百炼 API Key。
3. 编辑项目根目录中 Git 忽略的 `.env.web_search.local`，填写 `BRIGHTDATA_API_KEY`、`BRIGHTDATA_ZONE` 和 `WEB_SEARCH_DASHSCOPE_API_KEY`。不要把真实密钥填入 `.env.example`。
4. 如需微信公众号搜索，在同一私有配置文件填写 `SEARCH1API_API_KEY`。

Search1API 当前官方计费口径是：普通 `/search` 成功请求固定1 credit，与实际返回1条或3条无关；API参数允许 `max_results` 最高50。设置 `crawl_results=n` 后，总费用为 `1 + 成功抓取正文的页面数`，抓取失败的页面不额外计费。因此本项目每一路3条候选 + 最多3条正文，单路通常为1–4 credits；三路并发且9篇正文均成功时最多12 credits。失败状态不扣费，成功但零结果仍扣1 credit。实际余额可用免费 `/usage` 接口前后核对。
4. 保存后下一次搜索会读取新密钥，无需改动 Python 文件。已在运行的应用也会在下一次调用时重新读取该文件。

主地址为 `https://api.brightdata.com/request`，请求目标固定为 Google 搜索页，使用 `parsed_light` JSON；备用 MCP 地址为 `https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp`。两边都使用 Bearer 鉴权。搜索连接设置 `trust_env=False`，不继承 Windows 系统代理或代理环境变量；这不控制系统级透明转发或 VPN。`fetch_webpage` 和浏览器工具各有自己的连接设置。

Worker 的首轮发现调用可同时提交 1 个主查询和最多 2 个互补查询，Harness 在同一次工具调用里并发执行。每个查询只向模型展示前 3 条，最多展示 9 条；搜索端这一次实际返回的其余有效结果全部写入 Worker checkpoint，不进入模型消息。历史 Trace 中 Google 单次响应实际出现过 9 条和 10 条，百炼旧请求通常为 5 条；实现不假设固定返回数，而是按真实响应缓存。后续把 `next_page_ref`（或原始完整 query）放进 `continue_refs`，每次从本地缓存继续显示 3 条，`external_requests=0`。缓存读完后必须换更精确的查询重新发现，不会静默发起远程“下一页”。

`Q1/Q2/Q3` 只标识搜索分支，不能作为证据。每条真实返回结果带独立 `source_ref`：首批三路分别是 `WA1–WA3`、`WB1–WB3`、`WC1–WC3`，缓存翻页继续成为 `WA4–WA6`；同一Worker后续新查询从 `WD…` 继续，避免碰撞。Worker schema可直接引用这些W编号，Harness会验证它确实来自某次已完成的 `web_search`，只把被引用的单条结果交给Reviewer。产物仍使用 `A`，普通工具调用仍使用 `E`，验收条件仍使用 `C`。底层保留旧的单查询分页函数供兼容测试使用，但 Worker 注册的是上述批量发现与本地缓存接口。返回值中的 `backend` 表示实际后端；发生回退时还包含 `fallback_from` 和不含凭据的失败原因。两个后端都失败才返回整体错误。

2026-09-18 已用当前密钥通过项目 `web_search_tool` 完成一次真实查询：`Python 3.13 documentation` 返回 5 条结果，均为 Python 官方文档页面。原始工具结果保存在 Git 忽略的 `.agent/web-source-probes/20260918T115848_e2f2d842.json`。这证明当前账号已可调用该服务，但一次查询不足以衡量长期稳定性或结果质量。无密钥的直连检查得到 HTTP 401；连接远端为 `39.96.198.249:443`，未连接本机 `127.0.0.1:7897` 代理入口。

2026-09-19 已通过项目 `web_search` 实测主备链路：正常查询由 `brightdata_google_serp` 返回 Google 结果；人为注入主后端失败后，同一次工具调用由 `bailian_web_search_mcp` 返回 5 条 Python 官方文档结果，并带 `fallback_from=brightdata_google_serp`。该验证没有调用模型或运行 AppWorld。

接入依据：[Bright Data SERP API 文档](https://docs.brightdata.com/cn/scraping-automation/serp-api/query-parameters/google)、[阿里云百炼外部调用文档](https://help.aliyun.com/zh/model-studio/mcp-external-calls)。
