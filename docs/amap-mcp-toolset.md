# 高德地图 MCP 工具组

项目优先读取 Git 忽略的 `.env.amap.local` 中的 `AMAP_MAPS_API_KEY`，并连接高德官方个人 Key MCP。没有配置个人 Key 时，才回退到百炼高德地图 MCP 地址 `https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp`，复用项目根目录 `.env` 中的 `DASHSCOPE_API_KEY`。两种连接都设置 `trust_env=False`，不继承系统或环境变量中的代理设置。个人 Key 不写入代码、文档、Trace 或日志。

启动顺序：`ConversationRuntime.start()` 创建真实工具池 → 尝试连接高德 MCP 并读取工具清单 → **15 项约定工具全部存在**才加入工具池 → `ToolsetRegistry.build_router_metadata()` 依据真实工具池构建能力目录。缺密钥、连接失败或工具不齐时，高德四个能力组均不注册。下次启动会重新检查。当前设计没有后台健康轮询；运行中断线时，本轮调用会报错，重新启动后会重新判定可用性。

高德15项工具按共同调用场景拆成四个普通非互斥组：`AMAP_POI`（地点与周边）、`AMAP_ROUTE`（路线与距离）、`AMAP_LOCATION_CONTEXT`（地理编码、IP粗定位与天气）和 `AMAP_ACTION_LINKS`（个人地图、导航与打车入口）。它们可与 `WEB_RESEARCH` 等普通组组合；Worker 路由最终最多选择三个组，Skill 声明的必需组会确定性并入。`APPWORLD` 仍是隔离评测环境的互斥组。

能力包括地点与坐标、周边地点、距离、驾车/公交/步行/骑行路线、天气，以及地图或导航链接。用户未提供当前位置时不会假定已获取设备定位；附近餐馆列表也不等于口碑评价。

2026-09-18：使用百炼密钥进行真实连接，返回 15 项工具；调用 `maps_geo` 查询“上海站”得到非空、非错误结果。后续仅改变模型可见分组，没有改变高德 MCP 的完整连接校验。

2026-09-20：公共试用 Key 的地点搜索返回 `USER_DAILY_QUERY_OVER_LIMIT`，而天气工具仍可用。项目因此加入个人高德 Web 服务 Key 优先路径；离线检查覆盖个人 Key 优先、百炼回退、工具齐全、缺失一项和缺密钥。
