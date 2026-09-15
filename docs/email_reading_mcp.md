# 本地只读邮箱 MCP

PersonalOps 可选接入 `@ethanli666/qqmail-mcp@1.2.1`，用于读取 QQ 邮箱、Foxmail，以及可通过标准 IMAP 连接的其他邮箱。本机使用单独的 `.env.email.local` 保存账号配置；授权码未填写时不会启动子进程，也不会向模型注册邮箱工具。

## 能力边界

- 可检查连接、列出近期邮件、读取短摘要/正文、查看附件清单，并下载用户指定的附件。
- 附件只保存到 Git 忽略的 `.agent/email-attachments`，使用清理后的文件名且不覆盖已有文件；系统不会自动打开、执行或解压。
- 可通过独立的本地 `email_create_draft` 工具向配置的 Drafts 文件夹追加纯文本草稿。
- 不存在发送、回复、删除、移动、标记邮件的工具；草稿返回值明确包含 `sent: false`。

这个边界不是 Prompt 约定：上游只使用只读 IMAP 邮箱锁，PersonalOps 又使用精确工具白名单。附件下载只写本机文件，不修改邮箱；即使上游以后新增发送等邮箱写工具，也不会自动进入工具池。

## QQ / Foxmail 配置

先在邮箱设置中启用 IMAP，生成专用授权码（不是登录密码），然后只在本机 Git 忽略的 `.env.email.local` 中填写：

```dotenv
EMAIL_MCP_ENABLED=true
EMAIL_MCP_ADDRESS=你的邮箱地址
EMAIL_MCP_AUTH_CODE=你的IMAP授权码
EMAIL_MCP_IMAP_HOST=imap.qq.com
EMAIL_MCP_IMAP_PORT=993
EMAIL_MCP_FOLDER=INBOX
EMAIL_MCP_DRAFTS_FOLDER=Drafts
EMAIL_MCP_SECURE=true
```

项目已经把开关预设为 `true`；但在地址和授权码为空时会安全地保持未连接，不影响其他能力启动。填好两项并重启 PersonalOps 后，启动过程才会建立本地 STDIO MCP 会话。日志不会输出邮箱地址或授权码。

## 其他国内邮箱

只要服务商提供标准 IMAP，就可以替换 `EMAIL_MCP_IMAP_HOST`、端口和默认文件夹。不同服务商的授权码开通方式、IMAP 主机名及风控规则不同，应以服务商当前官方文档为准。

## 草稿为什么不会被发送

已审计的上游包仍保持纯只读。草稿由本地主机的独立工具完成，只构造纯文本 MIME 邮件并对 `Drafts` 执行 IMAP APPEND；项目没有导入或配置 SMTP 客户端。QQ 实际文件夹清单已经确认存在 `Drafts`。保存成功只表示草稿可在邮箱中看到，不表示已经投递。
