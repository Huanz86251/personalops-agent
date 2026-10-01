# 文件通路诊断（2026-09-06）

结论：不是全链路卡死。Web专用下载→本地解析→内部发布→General/Code读取可用；邮箱下载与浏览器点击下载缺少任务文件登记。PDF失败还可能来自取错内容、大小上限、分页/OCR或模型附件能力，不能统一归为沙箱读不到。

本轮只诊断，新增显式探针与记录，未修改生产运行时、Skills、权限、镜像或模型配置。没有收费模型、发送、草稿写入、真实飞书网络请求或服务启动；复用上一轮已授权下载的邮件附件，不再查看邮箱。项目其他任务对记忆/飞书卡片的修改保留。

## 1. 真实角色边界

| 角色 | 运行/文件空间 | 自动可见范围 |
| --- | --- | --- |
| Web Worker | 宿主上的任务图；默认StateBackend，/handoff路由至运行目录的只读FilesystemBackend；浏览器另有租约 | 本Worker.files、登记给本Worker的下载引用、同run已发布/handoff |
| General Worker | 同样是任务状态文件空间，不是Code的Docker | 自己的files与同run/handoff；不自动获得Web私有files或下载记录 |
| Code Worker | Docker，/workspace可写，/handoff只读挂载 | 挂载的已交接文件和工作区，不可直接读宿主邮箱/browser目录 |
| Code Reviewer | 独立Docker，/workspace只读、/review可写、/handoff只读 | Worker候选与已交接文件，不共享Worker私有/tmp |
| Step Reporter / Scheduler | 有界证据/结构化事实，不是共同文件读取器 | 被提交的证据、审核包与发布回执；不自动看到全部附件原文 |

代码入口：`run_workspace.py:create_run_worker_backend`、`workers/general_runtime.py`、`workers/web_runtime.py`、`workers/docker_sandbox.py:create_pair`。共享不是“所有Agent自动读所有下载”：跨角色必须显式产生可复用产物并按既有宿主发布流程交接，防止私有文件和跨任务数据串用。

## 2. 已跑通与断点矩阵

| 来源/操作 | 已验证 | 结论 |
| --- | --- | --- |
| Web download_web_artifact | 真实WebStepRuntime和工具通过本次自建HTTP服务下载两页PDF，状态自动登记worker_downloaded_artifacts | 同Worker可以用返回的reading_path调用attachment_to_text |
| read_file直接读/downloads引用 | 真实图返回File not found | /downloads是附件读取器处理的虚拟引用，不是通用文件系统路由；应使用工具指定的读取方式 |
| 解析结果/artifacts/read.md | Web真实read_file回读含Native heading | 原生工具输出确实进入Worker.files，不只是回复里打印路径 |
| 跨Worker直接挪私有下载记录 | 即便复制记录，其他worker_id读取仍被所属目录检查拒绝 | 正确隔离，不是该扩权的故障 |
| Web→内部发布→General | 真正resolve候选、publish_artifact_to_handoff；GeneralStepRuntime读PDF第2页及Web生成的Markdown | 主干可用；本探针人工决定发布候选，未测试模型自主审批质量 |
| 内部发布→Docker Worker/Reviewer | 两个真实容器读取PDF与Markdown，SHA256逐项一致；写/handoff均被拒 | 文件交接和只读权限有效；本次容器/卷均已清理 |
| 飞书附件→持久化恢复→General读取 | 现有验收在真实SDK消息/SQLite/队列/GeneralRuntime/本地OCR上通过，通道下载是替身 | 已有受控导入实现；不冒充真实飞书网络测试 |
| 真实邮件附件 | 本轮复用已下载751199字节PDF，手工导入后本地OCR读完1页，无warnings | PDF可解析，但邮箱生产下载结果仍无files、/downloads或/handoff登记 |
| 浏览器点击下载 | 本次真实项目MCP从自建页面点击附件，文件实际落盘且字节匹配，返回list而非状态Command | 无Worker下载登记；本机路径被拒，伪造/downloads也被拒；第二个已复现导入缺口 |

Web图用本地脚本模型发出实际工具调用，不接模型供应商；浏览器租约在该图测试中是空工具替身，浏览器下载另用真实Playwright MCP测试。以上不是完整自主任务成功率或所有路径一次统一端到端测试。

## 3. PDF问题分层

1. **工具选错**：真实访问W3C公开dummy.pdf，fetch_webpage拒绝application/pdf并提示走附件流程；改用真实下载工具取得13264字节、SHA256 `3df79d34abbca99308e79cb94461c1893582604d68329a41fd4bec1885e6adb4`，attachment_to_text读到“Dummy PDF file”，1页、无warning。不是PDF完全不可读。
2. **下载内容不是真PDF**：自建端点/login.pdf返回HTTP200登录HTML，下载状态仍是DOWNLOADED，解析器拒绝Unknown or unsupported file content。下载成功只表示收到了字节，不能批准内容完成；应对Content-Type/文件签名作校验，登录页走访问条件缺口，不让Code安装PDF依赖来修。
3. **容量不同**：Web默认下载上限100MiB，读取器MAX_BYTES=20MiB；真实下载约21MiB测试文件成功，读取明确拒绝超过20MiB。输出本已提示此差异，但Scheduler仍需按读取能力规划；不可反复重新下载。飞书单文件上限20MiB相对一致。
4. **部分读取**：两页PDF只请求第一页返回partial、next_page=2、明确警告；General读第二页返回next_page=null但仍注明只读第2页。next_page为空不等于此前所有页已读，需累计覆盖。默认最多20页/次，不应一页成功就宣称全文。
5. **扫描内容/OCR**：真实邮件PDF先前ocr=off有未读图像警告；本轮auto实际识别一张图并读完1页。OCR进程释放；66项回归还覆盖中英文、混合PDF、DOCX/XLSX嵌图、超时进程回收、引擎失败保留原生文字并提示。不保证OCR准确率或图表理解。
6. **沙箱依赖**：本次Docker中pypdf、fitz、rapidocr_onnxruntime均不存在。原始文件在/handoff可见不等于Code的Python可以解析。优先让General/Web用现有本地读取器产出带页码Markdown/JSON，再发布给Code，不为读文件扩大Docker宿主权限。
7. **read_file原始附件与模型能力（潜在运行故障）**：本地DeepAgents实现对PDF返回base64文件内容块，并不自动提取文字。当前deepseek-v4-flash的SDK profile有attachment=false，却没有pdf_inputs字段；实际调用本地过滤函数后file块仍保留，因为缺字段默认允许。未调用供应商，故不声称已复现HTTP报错；但工具“读取成功”不能证明当前模型理解PDF，必须补能力映射/预检或走文字提取。其他支持PDF的模型可能有不同结果，不能一概禁止所有模型原生PDF。
8. **网页编码不是PDF故障**：电影博物馆影讯属于HTML，先前HTTP缺少字符集时工具按UTF-8解码导致乱码；同页浏览器可读。本轮项目fetch再次取得乱码（`.agent/web-source-probes/20260906T072908_27a47aee.json`）。当前fetch_webpage仍用response.encoding直接decode，未按HTML meta判断。这应归网页解码层，不归PDF解析或文件挂载层。

公开PDF来源：[W3C测试文件](https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf)。此次实际下载留在 `.agent/web_worker_artifacts/worker-07c9257135192c82fce7/downloads/3df79d34abbca993-dummy.pdf`。

## 4. 另发现的边界与重试风险

- **公开下载的目标校验不一致**：fetch_webpage拒绝本机/私网；download_web_artifact只检协议/URL凭据且自动跟随重定向。本轮只访问自建127.0.0.1服务，下载和重定向均成功，没有探测其他本地服务。若该工具承诺仅公开网络，应对初始地址及每跳重定向应用一致边界；这是潜在SSRF入口，不是应该为邮箱导入开放的捷径。
- **浏览器会话与HTTP下载分离**：download_web_artifact独立建立httpx.Client，没有接入浏览器cookies。匿名公开下载成功不能推断需会话的地址也能下载；不要复制凭据给无关工具。浏览器已实际取得的文件应由受控导入处理，而非强制重新请求一遍。
- **邮箱重试不是幂等复用**：对已缓存固定版本上游saveAttachment做离线真实磁盘试验，同一邮件UID/part/文件重试得到Attachment already exists且原内容保留；不同part的同名附件会加part前缀，能正确分开，不是同名就冲突。目录键未含账号身份或UIDVALIDITY，切账号/重建邮箱时存在身份冲突风险（源码推断，未切换真实账号）。
- **普通本机文件**：不能因用户说出宿主路径就假定已进入任务空间。文件选择/授权与受控导入是必要步骤；当前文件阅读器拒绝任意宿主路径是正确边界。

## 5. 邮件应该怎样修（尚未实施）

优先做统一的“任务文件导入”层，让邮箱和浏览器复用现有可读引用/发布机制，而不是把所有下载目录挂进每个沙箱。

1. **源回执到受控记录**：在真实邮箱/浏览器工具调用完成后，由宿主包装器接收回执；仅允许读取该邮箱或该浏览器租约的受控根目录，检查实际路径、大小、哈希、类型，拒绝任意路径/符号链接越界。
2. **分离来源身份与内容身份**：邮箱来源键用账号指纹、folder、UIDVALIDITY、UID、part；浏览器用租约和下载ID；内容用SHA256校验。秘密不写日志，不将同文件名当身份。
3. **当前Worker立即能读**：包装器返回Command，把运行归属、Worker归属、持久化位置登记到任务文件账本，并返回明确的reading_path、大小/类型、支持的读取方式。存储完整文件，不把大块base64塞入模型上下文。
4. **跨角色按既有交接走**：Worker读取/加工后提交真实候选；按General自报告或Web独立审核的既有宿主流程发布/handoff。Code接收只读原始文件及必要的Markdown/JSON；不自动共享整个邮箱或其他run文件。
5. **恢复与重复调用**：持久记录需在checkpoint/重启后可验证恢复；同源重复下载先核对旧文件身份和哈希再复用，不覆盖、也不无限重试；缺失或变更必须明确失败。不同账号、run、Worker之间不能借同名路径越权。
6. **分别验收**：下载字节、登记可读、解析范围、跨角色可读、Scheduler收到有出处事实各自有证据。没有附件全文进入Scheduler不算故障；缺的是Worker真正读过并提交必要事实，而不是把全部文件灌给调度模型。

建议优先级：统一邮件/浏览器导入与公开下载地址边界→模型PDF能力映射→容量/格式预检、重试幂等→网页编码修正。验收至少包括邮件重启恢复、浏览器会话下载、跨run拒绝、PDF分页/OCR、大文件/HTML伪装，以及真实Docker内读交接文件。

## 6. 可复现证据与边界

- `scripts/probe_file_routes.py --docker`：`.agent/file-route-probes/20260906T072049_ac2ce3fe/`，包含Web/General真实工具结果、发布回执、Docker双方字节核对和清理证明。前两次仅探针参数/脚本模型签名错误已改，未改生产Schema放宽测试；失败目录保留。
- `scripts/probe_browser_download_boundary.py`：`.agent/browser-download-probes/20260906T072238_f1540d9c/`。4次真实MCP调用，浏览器文件已保存但无状态登记；会话及自建服务正常关闭。
- 邮件已有附件复查：`.agent/email-read-probes/20260906T064726_3b3e687f/handoff-072250-a00f6e/`。含私人内容，仅Git忽略存储，不公开。
- `scripts/probe_email_attachment_collisions.mjs <已验证的上游src/index.js>`：`.agent/email-collision-probes/149b922f-e7f0-4bf3-a521-c29fd54dc0ad/`。合成内容，无IMAP调用。前次探针把不同part成功标签错误写成UNEXPECTED_SUCCESS，复查后修正为SAVED且distinct_path=true，未更改上游行为。
- 回归：Web下载、飞书附件、OCR、原生文件工具、发布、提交、GeneralRuntime、WebRuntime共66项通过（44.11秒）。测试使用网络/模型替身的部分如上说明；有现存SDK弃用警告及pytest清理旧临时目录的Windows警告，不代表业务测试失败，未手工清理其他任务临时目录。
- 仅新增诊断脚本和文档。只清理本次2个Docker容器/2个卷，合成文件、真实邮件附件和公共PDF保留；无commit/push。
