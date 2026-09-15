# APPWORLD 能力卡

## 选择它

任务明确运行在 AppWorld 评测环境，需要通过 appworld_discover 查阅模拟应用 API，再用 appworld_execute 操作其中的邮件、日历、文件、购物、联系人等模拟应用来完成任务。

## 不选择它

- 操作用户真实邮箱、真实浏览器、真实文件或线上服务。
- 普通 Python 开发、测试或本地 Shell 工作。
- 任务没有同时提供 appworld_discover 与 appworld_execute 工具。

## 典型命令

- 在 AppWorld 中查找联系人并发送模拟邮件。
- 使用 AppWorld API 更新模拟日历事件。
- 先查阅应用 API 文档，再完成这道 AppWorld 任务。

## 易混淆边界

APPWORLD 是隔离评测世界，不是 BROWSER_AUTOMATION，也不是用户真实账户。只有真实工具池中同时存在 appworld_discover 与 appworld_execute 时本组才会成为候选。
