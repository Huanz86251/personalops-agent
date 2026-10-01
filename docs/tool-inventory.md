# 当前工具清点

> 2026-09-06 补充：下文 22 项是 9 月 5 日快照，不含后来注册的日程、邮箱及 `capture_desktop_screenshot`。当前声明见 `tools/schedule_tools.py`、`tools/desktop_tools.py`、`mcp_runtime.py`、`toolsets.py`；实际可用仍取决于配置、启动和角色过滤。邮箱无发送能力。对应策略及限制见 [场景 Skill](skill-scenarios.md)。

最新桌面观察能力只有一项原生截图工具：按用户明确要求把当前 Windows 桌面保存为当前任务 `/artifacts/*.jpg`，可继续交给 `ocr_image`。它没有点击、输入、持续录屏或任意宿主文件浏览能力，也不会自动发飞书。2026-09-06 项目自身真实冒烟取得 3840×2160、207951 字节 JPEG，并由本地 OCR 成功生成 Markdown；没有使用外部视觉模型或网络。

2026-09-05，按当前工作区代码核对；不以 GitHub 旧提交代替本地未提交代码。
此次没有删除旧工具，没有启动 MCP 服务来探测线上工具，也没有修改动态 Skills。2026-09-06 工具组选择已接入当前规划入口并改为常驻 Cross-Encoder；工具清单本身不因此获得新权限。
可运行 `python scripts/tool_inventory.py` 输出注册工具完整 schema、角色及条件，方便后续管理器接入。
此脚本只检查声明，不调用模型、浏览器或飞书。

## 原生注册工具（ALL_TOOLS，22 个）

| 名称 | 当前入口/状态 |
|---|---|
| get_current_time | General / Web |
| attachment_to_text | General / Web，文档阅读，内含按需 OCR |
| ocr_image | General / Web，中英文图片文字提取 |
| convert_document | General，格式转换 |
| spreadsheet_read | General，表格读取 |
| spreadsheet_write | General，表格写入 |
| spreadsheet_format | General，表格格式 |
| spreadsheet_chart | General，Excel 内图表 |
| symbolic_math | General，SymPy |
| python_syntax_check | General，语法检查，不执行提交代码 |
| python_static_check | General，Ruff 静态检查 |
| web_search | Web |
| find_github_mirror | Web |
| fetch_webpage | Web，网页正文 |
| host_list_files | General / Web，仅在Step声明主机文件访问时分页发现允许根目录的一层名称与元数据 |
| host_search_files | General / Web，仅在Step声明主机文件访问时按名称递归搜索允许根目录 |
| host_import_file | General / Web，精确路径和理由经独立Judge／必要时Human Loop后只读导入私有 `/downloads` |
| send_local_file_to_feishu | 仅已配置且验证身份的 General 任务；提交路径与理由，独立Judge存疑时由用户允许或拒绝 |
| request_additional_capabilities | General/Web/Code Worker 固定可见的一次性控制工具；用 U1/S1/C*/E* 证据申请缺少的能力类别，由独立选择器从当前角色可执行的剩余能力组中补选，第二次申请强制按已有证据收尾 |

General 的文件业务工具处理任务文件系统；新回传工具处理明确允许的宿主机绝对路径。
二者不是同一个路径权限范围。Code 主运行时传入空业务工具列表，使用 Docker 文件/执行工具。

## 框架注入工具

- General/Web 文件工具：`ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`。
- Code Worker/Reviewer：以上工具，加 `delete`、`execute`；执行后端为各自 Docker sandbox。
- General 完成：`report_general_result`。
- Web：`download_web_artifact`、`publish_worker_progress`、`complete_web_step`。
- Code Worker：`publish_worker_progress`、`submit_code_for_review`、`respond_to_code_review`、`submit_continued_code_for_review`。
- Code Reviewer：`request_code_worker_repair`、`publish_reviewed_candidate`、`submit_code_review`。
- Web 仍由 Deep Agents 默认注入 `task`，本轮用离线模型实际检查首次绑定列表确认存在。
  General 不提供它；Code 的工具白名单过滤它。这里仅记录，不在本轮调整 Web 委派流程。

工具数量不能只数 ALL_TOOLS；工厂、中间件、资源租约还会增加工具。
控制协议工具也不能与普通办公工具采用完全相同的权限处理。

## Playwright MCP

项目已有 `mcp_runtime.py`，Web 调用时从浏览器池租用实例，加载服务实际返回的工具后取白名单交集。
当前白名单 11 项：

`browser_navigate`、`browser_navigate_back`、`browser_snapshot`、`browser_find`、
`browser_click`、`browser_type`、`browser_fill_form`、`browser_select_option`、
`browser_wait_for`、`browser_tabs`、`browser_close`。

当前不允许 `browser_evaluate`、`browser_run_code_unsafe`、`browser_file_upload`、
cookie/localStorage 写入工具。服务版本不一定提供白名单里的每个名字，实际可用列表需在真实租约建立后核验。
本轮未新装或启动其他 MCP；若用户说的 Freewrite 指另一个 MCP，其工具尚不能算已接入。
浏览器点击/填表本身仍可能产生外部副作用，后续分级不能简单把整个 Web 组标成只读。

## Shell 和兼容代码

旧 `middlewares.build_middlewares()` 仍包含 `ShellToolMiddleware` 和 `ToolsetRouterMiddleware`；
当前源码搜索没有找到该构建函数的生产调用点，因此宿主机 Git Bash `shell` 并未挂到当前入口。
现用代码执行是 `workers/docker_sandbox.py` 的 Docker `execute`，并未丢失。

旧 ToolsetRouterMiddleware 复用新的 Cross-Encoder 路由器作为兼容路径；当前生产入口则在 Conversation Runtime 准备规划上下文时调用同一路由器。旧生成式路由提示词已经删除。旧宿主文件工具已经从注册表和源码移除；当前 Worker 只使用各自后端提供的 `ls/read_file/write_file/edit_file/glob/grep`。旧宿主 Shell 仍可作为后续独立清理候选。

## 评测专用和后续分级

`appworld_discover` 与 `appworld_execute` 仅在 AppWorld 评测环境注入；前者只查接口文档，后者操作模拟业务数据；不等于日常助手获得真实邮箱、账户或主机权限。
当前没有日常邮件发送/搜索、云盘上传、全盘相似文件搜索、远程应用启停工具。

下一阶段可按每个工具记录：角色、读写范围、外部副作用、可用条件、审批要求、加载方式。
先基于这份清单讨论，再做全工具统一分级/动态暴露。当前只为新增回传工具实现强制身份与确认边界。

