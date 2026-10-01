# 桌面截图、OCR 与文件回传

## 已实现：项目自己的截图与 OCR

`capture_desktop_screenshot` 是 PersonalOps 进程内的 LangChain 工具，不是 Codex、ChatGPT 或其他宿主自带的 computer-use 能力。它使用本机 Pillow `ImageGrab` 截取主显示器或整个 Windows 虚拟桌面，压缩为 JPEG，并写入当前 General Worker 的 `/artifacts` 状态。模型参数只有输出路径、是否包含所有显示器和 JPEG 质量；`ToolRuntime` 隐藏注入。

截图仅在用户明确要求时使用，不持续监控或后台连拍。输出最大 20 MiB，不写 `.agent` 临时宿主路径，不自动上传。需要文字时把返回路径交给现有 `ocr_image`；OCR 返回文字、像素框和置信度，不生成场景描述。

2026-09-06 用项目工具自身做真实冒烟：主显示器为 3840×2160，JPEG 207951 字节，随后 `ocr_image` 成功生成 `/artifacts/desktop-ocr-smoke.md`。测试对象只存在于临时 Worker 状态，没有发布或上传。

## 是否必须使用多模态模型

当前截图与文字识别不需要多模态模型：

1. `capture_desktop_screenshot` 负责取得本地像素并保存任务 artifact。
2. `ocr_image` 在本地提取文字、像素框和置信度，结果交给文本模型总结。
3. 只有用户未来要求开放式理解图标、画布或复杂空间关系时，才需要另行评估视觉模型；这不属于当前能力。

因此当前不会把整张 4K 截图交给主模型，也不会因为有截图而暗示模型可以安全点击。

## 明确不接入桌面控制 MCP

按用户最终决定，PersonalOps 不注册 Windows UI Automation、鼠标、键盘或其他桌面动作 MCP；也不保留相关 Runtime 配置、工具 Schema、General Skill 或本地二进制。`DESKTOP_OBSERVATION` 只包含项目原生截图和本地 OCR。屏幕内容只能被观察，不能触发点击、输入、提交、应用启动或文件对话框操作。

## 当前飞书文件链路

`send_local_file_to_feishu` 处理宿主绝对路径、当前 `/downloads`、已审核 `/handoff` 和当前 Worker 工作区文件。必须同时满足真实飞书 owner/chat、当前已验证 Event 和逐次独立 Judge；Judge 存疑才在原会话询问用户。目录会先制作不可变 ZIP 快照；收到飞书 message ID 才算 `SENT`。

owner 或 chat 为空时工具不会暴露给 Worker，也不能发送 Workspace 或桌面文件。配置开启后：

- 用户给出 `E:/Downloads/report.pdf` 时，可以直接申请发送。
- 只说“桌面上那个文件”时，Worker 可先用只读目录工具定位，再申请发送。
- `/artifacts/screenshot.jpg` 等当前 Worker 文件可被冻结为不可变宿主副本后申请发送；这仍单独经过 Judge。
- Schedule 不直接读取文件。`schedule_create_agent_task` 到期后创建新 Event，届时 Worker 才获得该 Conversation 的任务环境；Scheduled SYSTEM Event 也不会凭空继承文件导出身份。

因此“截图 -> OCR”和“同一任务工作区文件 -> 独立审批 -> 飞书附件”都已接通；截图工具本身仍不直接上传。
