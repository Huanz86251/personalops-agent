# 本地文件发回飞书

2026-09-05 新增本地 LangChain 工具；2026-09-19 当前接口为 `send_local_file_to_feishu(path, reason)`。发送与读取分权：每次发送先由独立Judge核对用户原始请求、当前Step、精确路径和理由；存疑时暂停等待用户只回复“允许”或“拒绝”。
复用现有 FeishuChannel SDK，不增加 MCP Server，不启动新常驻进程。公开样例配置仍为 dummy；实际白名单仅在本机、Git 忽略的 `.env` 中配置。

## 配置接口

`config.Settings.feishu_export` 是 `ExportPolicy`，由以下环境变量加载：

```dotenv
FEISHU_EXPORT_OWNER_IDS=[]
FEISHU_EXPORT_CHAT_IDS=[]
FEISHU_EXPORT_ROOTS=[]
FEISHU_EXPORT_MAX_FILE_MB=30
```

前三个是 JSON 数组；`FEISHU_EXPORT_MAX_FILE_MB` 是正整数，单位为十进制 MB，默认 30。`.env.example` 中保留注释形式的 dummy 身份示例。`FEISHU_EXPORT_ROOTS` 仅保留为旧部署兼容配置，不再限制源路径；真正的外发边界是当前认证的 owner/chat、逐次独立 Judge 和存疑时的 Human Loop。
身份使用真实 `ou_...` 和 `oc_...`，不能使用昵称、模型推测或消息正文中的身份。
在飞书单独发送 `/file_whoami` 可以查询自己的用户和当前会话 ID；查询不会授予权限。
不要把整块系统盘或整个用户主目录配置成办公导出目录。

当前应用共享单个 owner 对话。配置 OWNER_IDS 后，入口会同时要求用户和会话在白名单，
阻止其他人进入这个共享对话；尚未改造成多租户应用。配置为空时保留既有入口行为，
但新导出工具不可见、不可执行。旧 `.env.example` 的 `FEISHU_OWNER_OPEN_ID` 没有当前代码读取点，
本功能实际使用上述新配置，不能误以为旧字段已提供验证。

## 本地文件回传

1. 用户可发送“把 E:/Downloads/report.pdf 发给我”，或指定一个目录；也可以要求发送当前 Worker 的 `/downloads`、已审核 `/handoff` 或工作区文件。路径可解析不等于获得外发授权。
2. 现有入口将真实 sender/chat 与事件写入附件登记库；运行/恢复时从登记库重建可信身份。
3. 已验证飞书任务可向 General、Web、Code Worker 提供此工具；Reviewer 不提供。Code 新生成的文件必须先经 Code Reviewer 发布到 `/handoff`，再由交付步骤发送。
4. 工具先经过独立Judge；明确匹配原始请求时继续，存疑时进入Human Loop。用户拒绝返回 `USER_DENIED`，不会伪装成文件不存在或工具失败。随后才解析虚拟引用，检查发送者、会话、文件类型与大小并创建不可变快照。
5. 宿主再次核对源路径及快照哈希，原子占用发送请求后上传、发送到当前白名单会话。
6. SDK 返回成功且包含 message_id 才记录 SENT，并将结果交给 General；不确定结果记 UNKNOWN，不自动重试。

可以用 `/file_status fx_...` 查询发送状态。历史审批命令仅作为旧请求的兼容入口，正常发送流程不使用它们。

## 大小与文件边界

- 普通文件原样发送；目录递归打包 ZIP，不能静默跳过失败项。
- 本机单次发送上限可用 `FEISHU_EXPORT_MAX_FILE_MB` 配置；实际生效上限取该值与飞书 IM 接口的 **30,000,000 字节** 中较小者。这是依据飞书接口限制采用的保守十进制边界。设为 `5` 会将本机上限收紧到 5 MB；设为 `500` 可以保存配置，但**不能**让当前飞书消息接口接收 500 MB 文件，生效上限仍为 30 MB。
  [官方 SDK 的文件参数说明](https://pkg.go.dev/github.com/larksuite/oapi-sdk-go/v3/service/im/v1#CreateFileReqBodyBuilder.File)
  标明 30 MB，空文件也不允许；[上传接口](https://open.feishu.cn/document/server-docs/im-v1/file/create)。
- 目录最多 100 个文件、扫描最多 1,000 个目录项、原始文件合计最多 100,000,000 字节；ZIP 仍须满足生效的单次发送上限。
- 超过生效上限时在调用上传接口前拦截，工具将明确报出字节数及由本机配置还是飞书平台限制，并把错误返回给模型；模型须告知用户未上传，并先询问是否允许拆分，不能自行拆分重发。
- 不接受 URL 或 UNC 网络共享；接受本机绝对路径、当前 Worker 的 `/downloads`、当前 run 已审核的 `/handoff` 和当前 Worker 工作区文件。
- 拒绝符号链接、Windows junction/reparse point、硬链接及 NTFS 附加数据流。
- 不再按目录名静态屏蔽源文件；独立 Judge 负责逐次判断是否符合用户原始请求，存疑必须询问用户。这不是完整内容 DLP，Judge 明确放行也不等于完成了敏感内容扫描。

## 授权和恢复

完整 schema 只有 `path` 与 `reason` 是模型参数，`ToolRuntime` 隐藏注入。
模型不能填写 recipient、owner、approved、approval mode；宿主白名单限定唯一用户和会话，独立 Judge/Human Loop 决定本次内容是否可外发，不依赖模型声称“用户已批准”。
工具元数据记录 capability、risk、request_role、executor、recipient、approval，作为后续统一分级的具体样例。
真正执行者仍是宿主发送服务；“模型较强”本身不是授权依据。工具只在已验证的 General/Web/Code Worker 中出现。

申请、状态、快照哈希、上传 key 和消息回执记录在 `.agent/feishu-exports/exports.sqlite3`。
快照在同目录 objects 下；成功、拒绝、过期后清理快照，过期清理在下次请求/相关操作发生时运行，
没有新增定时常驻清理进程。记录保留以去重，不包含文件正文。
上传前用 SQLite 条件更新原子占用请求；同一事件、源路径和内容的并发/恢复不会重复发送。
请求 UUID 同时传给飞书。崩溃停在 SENDING 或网络异常产生 UNKNOWN 时不自动重试；
需先在飞书核对是否收到，不能声称严格 exactly-once，也不能把 UNKNOWN 当作发送失败。
进程崩溃遗留的 SENDING 快照可能需要本地检查清理。

这是应用层防护，不是对具有相同操作系统账户权限的恶意进程的隔离。
不支持通过 Web/Shell 绕过导出白名单和大小限制。尚未对所有旧工具实现统一安全策略，不能把此工具的检查等同于全项目安全审计。

## 验证与剩余工作

离线测试覆盖真实 LangChain General 工具调用、真实 SDK 参数类型、SQLite 恢复，网络发送与主模型用替身。
原始回归覆盖未授权、错会话、错误角色、伪造事件/参数、目录/大小/凭据/链接拒绝、快照一致性、重复请求、超时未知结果；直接发送与超限报错已增加回归。
原 PDF 接收→事件恢复→General 阅读/OCR 回归保留。
本轮没有发送真实文件，没有启动或重启飞书服务，没有调用收费模型。
原有的 69 项组合回归是改为直接发送前的结果；本轮与文件回传、附件、菜单、工具门控及 General 链路相关的 59 项离线回归通过，包括直接发送、可配置上限、超限报错和不上传的测试。尚未发送真实文件，不能据此声称已完成在线验收。

尚未实现：飞书卡片按钮、云盘链接、大文件分片、全盘搜索/相似文件查询、文件哈希对比工具、
把任务虚拟产物直接转换为回传文件、全工具动态路由。当前需要用户给出确切本地路径。
