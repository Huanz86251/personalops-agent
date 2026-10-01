# 文件通路与离线 Code 镜像（2026-09-06）

本轮按用户要求修改运行实现，不把路径修复、解析分流、大小限制或发布权限交给 Skill / Scheduler 猜测。旧诊断保留在 `file-route-diagnosis.md`，其中的失败是修改前证据。

## 文件如何流转

```text
邮箱 / 浏览器 / Web 下载
  → 宿主校验来源目录、大小、SHA256
  → 当前 run + Worker 的私有副本 /downloads/<candidate_id>
  → read_file 本地解析，完整文字保存在任务文件中
  → Worker 提交文件候选及证据
  → 独立 Reporter（General/Web）或 Code Reviewer 审核
  → 宿主 Publisher 校验并复制至该 run 的只读 /handoff
  → 后续 Worker / Reviewer 按同一 handoff 引用读取
```

2026-09-21 主机文件边界调整为“本机只读默认开放、外发逐次审批”：General/Web/Code 可列出、按名称搜索、读取并把本机普通文件复制到当前 Worker 私有 `/downloads/...`，这些只读操作不再调用 Judge。默认范围是 Windows 当前已挂载的本地盘；服务端部署可把 `HOST_FILE_READ_SCOPE` 改为 `CONFIGURED_ROOTS` 并用 `HOST_FILE_ACCESS_ROOTS` 收窄。符号链接、目录联接、UNC 和 NTFS 附加数据流仍拒绝。

运行时会把当前开放的本机盘符或收窄后的根目录直接写入 `host_list_files`、`host_search_files` 和 `host_import_file` 的工具 description。Worker 因而可以直接使用 Windows 绝对路径。`host_list_files` 只列一层；`host_search_files` 按层广度优先搜索。二者达到扫描上限时返回已找到的子集和 `truncated=true`，不再扫描完整棵目录树后整次报错。

`ask_user` 是统一的人机回路工具，不再是 Scheduler 预写在 Step Schema 里的字段。Scheduler、Worker、Step Reporter、Code Reviewer、Replanner 与 Final Reviewer 在各自真正遇到决定性歧义时均可调用；Harness 暂停当前角色，飞书回答作为同一次工具结果追加在该角色上下文末尾，然后原角色继续。问答随后进入 StepReviewPacket 和 Scheduler 持久事实，供后续角色读取。它不等同于主机文件审批，也不会把答案先交给 Scheduler 改写。

读取/导入与发送是两个权限。已经读过文件不代表可以发到飞书；任何本机绝对路径、`/downloads`、`/handoff` 或 Worker 工作区文件在对外发送前，都由独立 `host_file_judge` 读取用户原始请求、当前 Step、Worker 身份、精确引用和理由。明确符合请求可放行，存疑时飞书任务暂停，用户只回复“允许”或“拒绝”。用户拒绝形成不可重试的 `USER_DENIED`，Reporter/Replanner不得把它写成文件不存在、工具缺失或普通 fallback，也不得换路径或工具绕过。

工作区之外的修改、覆盖、移动或删除不交给 Judge 自动放行，必须由用户明确允许；一次对父目录的同类操作批准覆盖该目录的后代。本轮没有向 Worker 注册这类破坏性工具，所以默认仍做不到修改宿主原件。

- `/downloads/...` 是虚拟引用，不是全局共享目录。新记录绑定 run_id / worker_id；私有磁盘副本放在 `.agent/task-files/<run-hash>/<worker-hash>/...`。原下载目录随后失效也不影响已登记副本。读取时重验大小与摘要；其他 run、Worker、任意宿主路径、篡改内容均拒绝。
- 普通 checkpoint 文件不能覆盖同名 `/downloads` / `/handoff` 的读取结果。浏览器只接受真实完成事件对应、且本次新增/变化的文件；路径必须位于该浏览器租约的输出目录。事件中虚构一个已有文件不能登记。
- 邮箱源下载目录按账号和运行时会话隔离，不跨会话直接复用旧 UID 对应的缓存文件。同一会话重复调用可复用成功回执，登记时仍校验文件；没有加入发送路径。
- General 纯文本结果继续走轻量自报；带文件的结果改走独立 Step Reporter。没有批准引用、审核预算不足或发布校验失败，都不能仅因 Worker 自报 COMPLETED 而共享文件。
- Reporter/Reviewer 负责判断，宿主负责真正复制与登记。模型填写一个公共路径不等于发布。`/handoff` 只是本次 run 内部共享，不是公网发布，也不等于对外发送；飞书等交付走独立 Judge / Human Loop 授权。
- Scheduler 接收报告、证据和发布引用，不自动接收所有私有附件正文。报告需要交代已读页、结论、限制；接收后续文件的 Worker 读取已批准的 handoff。

## PDF 与 read_file

此前 Deep Agents 的默认文件层可能把 PDF 返回为模型附件块。模型配置中的 `attachment=false` 不等于框架检查的 `pdf_inputs=false`，所以不能靠这一个标记保证安全分流。

现在 General/Web 的 `read_file` 对已登记下载和 PDF/DOCX/PPTX/XLSX 调用本地读取器；Code Worker/Reviewer 则在自己的离线容器中运行 `/opt/personalops/read_document.py`。返回文字，不让文本模型承担原始 PDF 解析。模型构造额外把不支持附件的 profile 映射到 PDF 禁用项作为第二层保护。

- PDF：offset 为起始页减 1，limit 每次最多 20 页；返回总页数、下一页和警告。扫描页自动 OCR。
- General/Web：沿用本机 RapidOCR 隔离子进程和其容量/超时限制；完整 Markdown 和详细 JSON 保存于 `/artifacts/`，模型只收到预览。
- Code：PyMuPDF 提取文字，Tesseract 简中+英文识别扫描/混合图片页；Office 提取基本文本，必要时可用 LibreOffice 导出 PDF。完整提取文字保存在容器 `/tmp`，预览截断不丢失全文。
- Code Office 文本读取不承诺复原布局、图片意义或重算公式。XLSX 上限为每表前 10000 行 / 100 列，超出会明确警告。复杂图表、视觉关系仍需另行验收。
- 有空页、OCR 失败、加密、超时或截断时，不能把“打开了文件”写成“完整理解了内容”。

## 统一边界

任务文件单个原始文件上限统一为 **20 MiB**：Web 配置默认/硬上限、邮件 MCP、私有登记、本地读取和飞书输入一致。入口可选更小上限，不能放大到读取器不支持的范围。Office ZIP 展开上限 100 MiB / 10000 项、网页正文 2 MiB、文字预览/分页属于不同资源预算，并不是原文件大小冲突。

浏览器 MCP 的底层下载由上游完成；超过 20 MiB 的文件不会进入任务文件空间，但这不是浏览器网络传输的硬字节闸门。通用文件工具采用有界流式下载。HTTP 200 登录 HTML 不再被当作成功的 PDF；每次请求及重定向都检查公开地址，拒绝私网/loopback。现有 DNS 检查与 HTTP 连接是分开的，不能把它宣称为已彻底防住 DNS rebinding 的网络隔离层。

中文 HTML 读取改为保留原字节，识别 HTTP 或 HTML meta 的字符编码；已补 GB2312 测试，避免默认 UTF-8 把开放时间/路线等字段读成乱码。

## 镜像内容与重建

新默认标签为 `personalops-code-sandbox:py312-v2`。直接 Python 依赖固定于 `docker/code-agent/requirements.txt`，构建内保存完整 `python-packages.lock`。包含：

- Python：pytest/coverage/asyncio/Hypothesis/Ruff、Pydantic/JSON Schema、NumPy/Pandas/SciPy/Matplotlib/SymPy、FastAPI/HTTPX 等。
- 文件：PyMuPDF/pypdf/pdfplumber/ReportLab、python-docx/python-pptx/openpyxl/xlsxwriter、Poppler、Pandoc、LibreOffice、中英文 Tesseract 及中文字体。
- 本地网页：Node/TypeScript/React/jsdom/esbuild/Prettier、Playwright 和 Chromium 二进制。
- 基础命令：Git、编译器、ripgrep、jq、SQLite、压缩与文件类型工具。

只在构建期联网安装；运行期仍 non-root、`network=none`、只读根文件系统、cap_drop ALL、资源上限及只读 handoff。下载依赖时国内镜像明显更快，本机采用构建参数 `DEBIAN_MIRROR` / `DEBIAN_SECURITY_MIRROR` / `PIP_INDEX_URL` 指向华为镜像，浏览器使用 `PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.npmmirror.com/binaries/playwright`；默认 Dockerfile 保留官方源。APT 签名及 TLS 校验未关闭。`.dockerignore` 只放行 Dockerfile、依赖清单和读取程序，不携带项目秘密。

后续源码有变化时需要重建，而不只是看到同名标签存在：`python scripts/setup_code_sandbox.py --rebuild`；本机网络可用 `--domestic-mirrors` 进行构建期镜像源选择。Python 安装和浏览器下载拆成独立缓存层，首次构建允许最多 60 分钟；不改变 Code 命令的运行时超时。

## 当前实测证据

- `.agent/runtime-file-bridge-probes/20260906T075357_323c6600/summary.json`：复用之前真实下载的邮件 PDF 回执，真实 General 图自动登记并经 read_file 本地 OCR 读完 1 页；无警告、无原始 PDF 模型块。不是重新访问邮箱，不伪称完整 IMAP 再下载实测。
- `.agent/browser-file-bridge-probes/20260906T075625_8262571d/summary.json`：真实项目 Playwright MCP + Web 图，自建页面点击下载，自动登记后 read_file 读到 `BROWSER_FILE_BRIDGE_SUCCESS`；浏览器及服务已关闭。
- `.agent/public-pdf-read-probes/20260906T080447_c8e6120a/summary.json`：真实 HTTP 获取 W3C 的 13264 字节 PDF，真实 Web 图直接 read_file 得到 `Dummy PDF file`；SHA256 `3df79d34abbca99308e79cb94461c1893582604d68329a41fd4bec1885e6adb4`。
- General 文件批准/未批准两种行为、跨角色/跨 run 拒绝、摘要校验、伪事件、HTML 假 PDF、重定向私网、GB2312 均有离线回归测试。结构化模型响应由测试桩提供，不是实际模型能力评分。
- Docker 双角色验证由 `scripts/probe_offline_code_documents.py` 执行，结果在 `.agent/offline-code-document-probes/`；下面的验收补记记录实际完成状态。

所有原邮件与内容证据均留在 Git-ignored `.agent/`。没有发邮件、写 Drafts、真实飞书发送、付费模型调用、批量评测、commit 或 push。

## Docker 验收补记

最终双角色证据：`.agent/offline-code-document-probes/20260906T083002_52af88c4/summary.json`。
最终镜像 `personalops-code-sandbox:py312-v2`：`sha256:551572cc849d40914fad986f4703e43637ef05959a198a210fe8bf255a44f4c0`，3,411,356,759 字节（约 3.18 GiB）。旧 v1 镜像保留。

- Worker：5 个真实容器内 pytest 用例通过（4.33 秒）；Reviewer：同样 5 个用例通过（2.78 秒）。覆盖数据/Schema 库与无效输入、外网 socket 拒绝、LibreOffice DOCX→PDF 实际文字回读、Playwright Chromium 按钮交互/截图、Node/jsdom/React/TypeScript、基础命令及简中/英文 OCR 字库。
- 两个真实角色图都通过 read_file 读共享 PDF 第 1 页，返回 `中文测试 销售金额 1200 元`、英文数字及原生标题；同样读到了 `/workspace/private.pdf` 第 2 页。Reviewer 使用自己的新模型上下文，不依赖 Worker 的隐藏对话。
- 两角色均为 uid 10001:10001、network=none；共享文件 SHA256 一致，写 `/handoff` 被拒。两个测试容器及两个卷在 Docker API 可达时逐个确认已不存在；原始证据留在 `.agent/`。
- 这是工具/容器/角色接线验收，采用有界脚本模型并在两次读取后结束，不是自主代码任务成功率评估，也不是一次真实模型业务审批。

实测带出的修复：PyMuPDF 1.28 的 `fitz` 旧别名会输出弃用警告，污染读取 JSON，已改用 `pymupdf`；Code 自然退出时通用收尾器曾强求未注册的 Web 终态工具，现按角色注册终止工具，Web 使用 `complete_web_step` 单次提交，Code 可选择初次/返修/继续提交。Docker 检查遇到连接错误不再当成资源不存在；清理命令失败也不再静默当成成功。WSL 刚唤醒时探针会有界等待既有 Docker 服务就绪。

回归记录：114 项文件/角色/配置相关测试通过；之后新增镜像源构建参数测试及角色收尾测试，针对最终相关改动的 39 项测试也通过。另 28 项 Code Agent/运行/发布/恢复测试通过；Docker 错误处理修改后的最终运行/发布/恢复回归 24 项再次通过，Reporter/Scheduler 证据交接 5 项通过（这些集合有重叠，不相加）。
