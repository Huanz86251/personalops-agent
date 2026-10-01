# 本地文档阅读与中英文 OCR

2026-09-05：已接入普通 LangChain 工具，没有新增 MCP Server、监听端口或常驻推理服务。
仅文字检测和识别，不使用图片描述/VLM，不调用付费模型 API。

## 模型与安装

- RapidOCR 3.9.2 + ONNX Runtime 1.29.0，CPU 执行。
- 飞桨 PP-OCRv6 Small 检测模型：`PP-OCRv6_det_small.onnx`，9,929,594 字节。
- 飞桨 PP-OCRv6 Small 识别模型：`PP-OCRv6_rec_small.onnx`，21,234,383 字节。
- 配套文字方向分类：`ch_ppocr_mobile_v2.0_cls_mobile.onnx`，585,532 字节。
- 检测＋识别权重合计约 31.2 MB；这是磁盘权重大小，不是运行内存。
- Small 支持简体中文、繁体中文、英文等；语言表以官方模型说明为准。
- 使用 wheel 内附的本地权重，显式指定路径。执行期禁用 Python 网络连接，不自动下载模型。
- 依赖和模型源码保存在 `.agent/ocr-deps/`；没有升级共享 Agent 环境。
- 安装：`C:/ProgramData/anaconda3/envs/agent/python.exe scripts/setup_ocr.py`。
  已存在目录时拒绝覆盖；`--verify-only` 只检查权重是否存在并更新本机指纹清单。
- 固定直接依赖见 `requirements-ocr.txt`；完整安装记录 `.agent/ocr-install.json`，
  模型 SHA-256 清单 `.agent/ocr-models.json`。安装到 Python 3.12 的依赖应由相同 ABI 的解释器运行。
- 安装曾因模型文件名顺序与预期不同而在安装后的检查阶段报错；已对照真实包修正并通过验证。
  pip 输出了共享环境 googleapis/protobuf 冲突提示；本实现使用 `-I -S` 独立子进程，
  只添加 OCR 依赖目录，不把这些包加入运行中 Agent 的导入路径。

官方来源：
- https://rapidai.github.io/RapidOCRDocs/latest/model_list/
- https://github.com/RapidAI/RapidOCR
- https://github.com/PaddlePaddle/PaddleOCR

## 两个模型可见工具

### attachment_to_text：真正的附件阅读入口

保留原名以兼容已有路由与调用，description 明确标注 READ、自动中英文 OCR。
无需模型先判断哪一页是扫描件。

参数：`path`、`output_path`；`ocr=auto|force|off`（默认 auto）、
`start_page=1`、`max_pages=20`（范围 1–20）。页码是 1-based，仅用于 PDF/PPTX。

- PDF：pdfplumber 提取原生文字并检查图片对象；正常文本页不加载 OCR。
  图片区域渲染后 OCR；文本过少/乱码且存在页面对象时渲染整页。force 强制整页。
  这是有边界的启发式，不是完整性证明；保留原生文字和 OCR 的各自来源，
  两者可能重复，不按相似字符串粗暴删掉可能重要的重复数字。
- PPTX：逐页提取文字、表格、备注，递归读取组合内图片；补读幻灯片关系中的
  图片填充/背景资源。图片裁剪应用于图片像素。OCR 内容带幻灯片和形状编号。
- DOCX/XLSX：原生文字由 MarkItDown 转换器读取，包内图片做 OCR 附录，
  以 `word/media/...` / `xl/media/...` 定位；不宣称恢复页码、单元格位置或阅读顺序。
- PNG/JPEG 等可解码静态图片：直接 OCR。GIF/TIFF 等多帧输入明确拒绝，避免默读第一帧。
- TXT/MD/CSV/HTML：原生转换；HTML 图片明确报告未读，不下载外部资源。
- 通过文件签名、Office ZIP 结构和实际解析识别类型，`.bin` 内的 PDF/图片可正常读。
  扩展名或 MIME 不是唯一依据；未知二进制文件不会送给 OCR。

### ocr_image：模型主动选图

参数：`path`、`output_path`，可选 `crop=[left,top,right,bottom]`。
裁剪坐标相对于应用 EXIF 方向后的图片，JSON 中的文字框相对于实际送入 OCR 的裁剪图。
PDF 应调用 attachment_to_text，不能把 PDF 字节当作图片解码。

两个工具均有参数描述、枚举/长度/数值范围 schema，`ToolRuntime` 不暴露给模型。
错误经过现有 ToolException 通道；部分识别失败保留可读内容并返回明确 warnings。

## 生命周期和资源

一次工具调用启动一个短期子进程，OCR 引擎在第一张需要识别的图片出现时才加载。
同一份文档本次读取的所有页面/图片复用一个实例；相同像素内容复用识别结果。
文档调用结束进程退出，释放模型；独立 ocr_image 在该次识别完成后立即退出。
没有跨工具调用常驻缓存，也没有后台 TTL 服务。分页后的下一次工具调用会重新加载，
默认每批最多 20 页；CPU 密集文档可以缩小范围。

- ONNX 计算线程 2，inter-op 1；OpenCV 线程 2；一次最多一个本地重工具。
- 输入 20 MiB、Office 展开 100 MiB/10000 项；图片最多 1200 万像素。
- 每次最多 60 张不同图片的实际 OCR；超限显式报告未读内容。
- PDF 默认至多 200 DPI，并在渲染前根据像素预算限制分辨率。
- 读取子进程硬超时 180 秒；超时后 kill/wait，回收进程并提示缩小页码范围。
- 同步工具被上层取消时，执行线程可能继续到本次完成或超时；不宣称即时取消。
- 这不是 OS 级恶意文件沙箱或硬内存配额。文件解析/推理均有成本。

## 文件访问与输出

输入可以是当前 Worker checkpoint 文件、当前 run 的只读 `/handoff/...`，或
当前 Web Worker 下载返回的 `/downloads/<candidate_id>`。下载文件需记录匹配、
所属 Worker 目录匹配且大小/SHA-256 未改变；不接受模型提供宿主绝对路径。

下载工具返回 `reading_path` 和阅读提示；Web Worker 可调用这两个阅读工具。
General Worker 同样可用。CODE 主运行时仍显式传入 `tools=()`，AppWorld 的显式
`appworld_execute` 工具集没有扩充；不是把宿主读取能力注入所有沙箱。

输出要求新建 `/artifacts/name.md`，同时生成 `/artifacts/name.json`：
- Markdown 按来源标注“文档原生文字”和“OCR 提取文字”。
- JSON 包含来源位置、识别文字、像素框、引擎分数、页面范围和限制。
- 工具消息返回最多 12000 字符预览及 reading 摘要、完整文件路径。
- 摘要保留 warning_count、前 12 条 warnings、next_page；详细 warnings 在完整结果中。
- `status=read` 仅代表所请求范围的读取无处理告警，不代表文档视觉内容已全部理解。
- 工件仍走现有 WORKSPACE_FILE 复核/发布流程，没有自动向用户发送。

## 已知能力边界

OCR 不描述场景、不理解图表趋势或箭头关系，不保证复杂表格行列结构。
评分低于 0.8 提醒复核，但高分也不保证准确；无文字结果不表示图像为空。
PPTX 母版/版式图形、SmartArt、原生矢量图表没有完整渲染，会告知限制。
PDF 对复杂向量文字/图形的启发式可能漏检，可通过 force 重读目标页。
DOCX/XLSX 附录可能包括未在正文显示的包内资源；外链图片不下载并报告缺口。
OCR 初次接入时仅支持任务中已有文件和下载文件。2026-09-05 后续已补
[飞书图片/PDF 附件接收入口](feishu-attachments.md)，通过当前任务的 handoff 供阅读。
接收入口改动了 main.py，复用既有事件队列；没有重启启动中的 Agent，需下次正常启动加载。

## 验证

`python -m pytest tests/test_local_ocr.py tests/test_local_native_tools.py tests/test_general_worker.py tests/test_general_runtime.py tests/test_web_runtime.py tests/test_web_artifacts.py -q`

真实 CPU OCR 样例覆盖中英混排图片、文字＋图片混合 PDF、扫描 PDF、双图 PPTX、
DOCX/XLSX 图片附录、下载工具到 OCR 路径；检查引擎复用、相同图片缓存、
纯文本不加载模型、进程退出、实际超时进程被回收、文件隔离和 schema。
没有调用外部模型 API、运行 AppWorld 题目或给出生产准确率结论。
测试图片使用 Windows 微软雅黑；其他机器需提供可用中文字体。

本次 41 项测试通过（29.74 秒）；后续仅对 PPTX 遍历做等价整理，重新验证了双图
PPTX 测试并通过。新增/修改的 OCR 核心文件通过 Ruff 检查和格式检查。
pytest 退出时报告了 Windows 既有临时目录清理告警，测试本身无失败；未清理其他任务目录。
