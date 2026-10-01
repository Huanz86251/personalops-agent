# 本地工具接入记录（2026-09-05）

后续 OCR 更新已完成：新增 `ocr_image`，`attachment_to_text` 升级为自动 OCR 阅读入口，
当前共 11 个本地工具。最新实现、生命周期和限制见 [local-ocr.md](local-ocr.md)。
下文保留基础接入记录；原先“附件没有 OCR”的限制已由该更新替代。

本次按用户指令加入 10 个普通 LangChain Tool。直接调用本地库，没有新增 MCP
Server、监听端口或常驻子进程。用户确认所说的 FastMCP 实际指 Fetch 网页读取。
没有安装 FastMCP 框架，没有改现有 Playwright MCP，也没有启动收费模型或评测。

## 本次能力

| Tool | 实现 | 运行方式 |
| --- | --- | --- |
| fetch_webpage | httpx + BeautifulSoup + markdownify | 进程内读取公开网页、转 Markdown、分段返回 |
| attachment_to_text | Microsoft MarkItDown 的原生格式转换器 | 本地 PDF/DOCX/XLSX/PPTX/HTML/CSV/TXT/MD 转文本 |
| convert_document | pypandoc_binary 中的 Pandoc | 按需短命令，30 秒超时；MD/TXT/HTML/DOCX 互转 |
| spreadsheet_read | openpyxl | 列出 sheet 或分页读取，返回公式而非重算值 |
| spreadsheet_write | openpyxl | 创建或修改 XLSX，最多 5000 个输入单元格 |
| spreadsheet_format | openpyxl | 字体粗体、数字格式、列宽 |
| spreadsheet_chart | openpyxl | 柱状、折线、饼图，图表嵌入 XLSX |
| symbolic_math | SymPy | 受限表达式的化简、展开、微分、数值计算、一元低阶多项式求根 |
| python_syntax_check | Python AST/compile | 检查语法，不执行代码 |
| python_static_check | Ruff | 按需短命令，10 秒超时，不自动修改、不执行代码 |

Fetch 是本地等价能力，复用开源 HTTP/HTML 库，不运行上游 Fetch MCP，也没有照搬
其全部功能。网页需要联网；其他功能使用本地输入，不调用收费转换或模型 API。
MarkItDown 按扩展名调用 upstream converter，不构造 Magika/ONNX 检测会话，不启用
插件、云端 OCR、图像描述模型或文档智能服务。原生库导入仍有一次性时间和内存开销。

## 源码、依赖与启动

- 项目接入源码：`tools/local_native.py`。
- 直接依赖版本：`requirements-local-tools.txt`。
- 本次单独安装目录：`.agent/local-tool-deps/`；Python 依赖的源码就在对应包目录。
- 安装记录及下载指纹：`.agent/local-tools-install.json`，保留本机，不作为公开工件。
- 安装入口：`C:/ProgramData/anaconda3/envs/agent/python.exe scripts/setup_local_tools.py`。
  目录已存在时拒绝自动覆盖；当前任务运行期间不升级依赖。
- 不修改共享 Agent 环境；需要依赖时追加搜索目录，已有环境的库优先。
  因此最终实际兼容性由真实转换测试验证，不能只凭单独目录的版本列表判断。
- 正常程序下次启动时加载新工具。本次没有重启服务，不强制热加载正在执行的图。

## 文件、资源与交付

输入是当前 Worker checkpoint 中的虚拟文件或本轮 `/handoff/` 文件。
不接受宿主绝对路径、URL 作为附件路径或 `..` 穿越。输出写入 checkpoint 的
`/artifacts/`，共享 `/handoff/` 只读。输出使用已有 `WORKSPACE_FILE` 类型提交复核，
由现有发布流程交付；文件生成不表示任务成功或已经发布。

二进制 XLSX/DOCX 使用 checkpoint 支持的 base64 编码，工具结果只返回位置和大小，
不要求模型复述二进制。文本转换保存完整结果，返回最多 12000 字符预览。
输入和输出各限 20 MiB，Office ZIP 展开总量限 100 MiB。重活最多同时执行一个；
忙时返回明确错误，不悄悄积压更多转换。Pandoc/Ruff 有命令超时；进程内转换和 SymPy
没有强制终止线程的超时保证，复杂文件仍可能耗时，不能声称完全没有 CPU 成本。

公开网页限制 2 MiB、最多 6 次请求，检查目标和重定向的公开地址，不用于登录页。
每次分段读取会重新抓取，尚无缓存。普通 DNS 检查不是抵抗 DNS rebinding 的网络沙箱。

## 限制与后续

- 表格工具写入公式，不运行 Excel 公式计算引擎；保留复杂已有工作簿的所有特性不作保证。
- Word 输出是 Pandoc 格式转换，不是复杂版式设计；本次未启用 PDF 输出引擎。
- 附件基础解析不等于扫描件 OCR；PDF/PPTX 等更多格式应按实际样本持续验收。
- Ruff 的未定义变量规则适合完整文件，不盲用于 AppWorld 持久会话片段。
- 网页工具只给 Web Worker。显式传入 `tools=[appworld_execute]` 的评测路径不自动扩充。
- 单位换算、图片处理、JSON 专用工具、Git/Gateway 和第 25—30 项优化没有在本次加入。

## 官方来源

- MarkItDown: https://github.com/microsoft/markitdown
- openpyxl: https://openpyxl.readthedocs.io/
- Pandoc wrapper/binary: https://github.com/JessicaTegner/pypandoc
- markdownify: https://github.com/matthewwithanm/python-markdownify
- SymPy: https://github.com/sympy/sympy
- Ruff: https://github.com/astral-sh/ruff
- Fetch reference (接口设计参考): https://github.com/modelcontextprotocol/servers/tree/main/src/fetch

## 验证入口

`python -m unittest tests.test_local_native_tools tests.test_general_worker tests.test_general_runtime tests.test_web_runtime`

覆盖真实 XLSX 创建/修改/图表/提取、MD→DOCX→文本、Ruff/SymPy、虚拟路径与跨 run
隔离、覆盖保护、工具分配、生成二进制的真实脚本模型图调用及已有工件解析流程。
所有脚本模型测试均不访问模型服务商。在线网页读取另做公开网页探针，不发送项目内容。

本次结果：24 项本地与 Worker 回归测试通过；真实访问 `https://example.com`
成功，获得网页 Markdown。没有运行 AppWorld 任务或收费模型；不宣称解题成绩提升。
