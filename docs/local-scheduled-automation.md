# 持久化日程与到期 Event

## 当前能力

`SCHEDULE_CREATION` 与 `SCHEDULE_MANAGEMENT` 共用同一套本地日程服务，不依赖外部日历。前者只暴露创建与立即通知，后者只暴露查询、暂停、恢复、取消和运行记录。日程定义和每次投递记录保存在 `.agent/schedules.sqlite3`，PersonalOps 运行期间由一个后台任务扫描到期项目，再按持久化的 `delivery_target` 分发。已有旧数据库启动时自动增加新字段，旧记录默认保持 `WINDOWS`。

当前工具：

- `schedule_create`：一次、每日或每周提醒；首次时间必须是带偏移的 ISO-8601 时间。
- `schedule_create_feishu_reminder`：一次、每日或每周向创建请求所在的飞书会话直接发送文本；不调用模型。
- `schedule_create_agent_task`：到期后创建一次新的 Agent `SYSTEM/QUEUE` Event；当前仅允许一次性任务。
- `schedule_list`：查看有效提醒或历史状态。
- `schedule_pause` / `schedule_resume`：暂停和恢复。
- `schedule_delete`：软删除，记录保留为 `CANCELLED`。
- `schedule_runs`：查看各投递目标的提交记录与错误。
- `windows_notify`：立即弹出通知，不创建日程。

用户说“明天、下周一、两小时后”时，Agent 必须先调用 `get_current_time`，再把含义确定为绝对时间。工具不会猜测无时区字符串。

飞书工具的参数中没有 `chat_id` 或 `conversation_id`。宿主根据当前工具调用的 `event_id` 从 Event Store 反查可信的会话地址，再把它随日程持久化，模型不能把提醒重定向给其他人。

## 到期后怎样进入 query

定时 Agent 任务不会修改某一条旧 query，也不会让 schedule 嵌套 schedule。数据流是：

```text
用户请求 -> schedule_create_agent_task -> SQLite Schedule
到期扫描 -> ReminderRun -> 新 AgentEvent(SYSTEM, QUEUE)
          -> SQLite EventStore + EventRun(QUEUED) -> EventRunPump
          -> 正常 Scheduler / Worker / 工具权限链 -> 飞书回复
```

因此到期任务和用户届时新发的一条普通消息处于同一持久队列。若当时已有任务运行，它按 `QUEUE` 等待，不会 `INSERT`、`REPLACE` 或打断当前任务。到期 payload 本身就是新任务的完整 query。它仍然受执行时的工具权限、用户确认和安全策略约束；提前创建日程不等于提前授权未来的高风险操作。

应用启动时 Schedule 扫描器先启动但保持投递门关闭。只有 EventRunPump 已启动且飞书长连接就绪后，主程序才注册飞书文本和 Agent Event 两个投递器并开门，避免应用半启动时把到期项目错误标成失败。关闭时先关投递门，再停止 Pump、飞书和底层 Store。

## 执行语义

- Windows 和飞书文本提醒到期后不调用模型；只有 `AGENT_EVENT` 明确要求届时执行任务。
- `SUBMITTED` 表示已提交给对应目标：Windows API 接受、飞书发送调用成功，或 Agent Event 已持久化入队；不等于用户已看到或 Agent 已完成。
- 进程关闭期间到期的一次性提醒，会在下次启动时补交一次。
- 重复提醒离线较久时只补交一个到期实例，然后跳到当前时间之后的下一周期，不制造通知风暴。
- 定时 Agent Event 使用由 `ReminderRun.run_id` 派生的确定性 `event_id`，进程在入队后、写入 `SUBMITTED` 前崩溃时，重启不会重复创建 Event。
- Windows 或飞书已经接受消息但本地尚未来得及写 `SUBMITTED` 时崩溃，重启恢复可能重复提交一次。这是当前偏向“不漏提醒”的 at-least-once 边界；若要对飞书做到严格 exactly-once，需要提供方幂等键或独立事务 Outbox/回执对账。

## Windows 启动边界

当前代码随 `ConversationRuntime` 启动和关闭，不会擅自修改 Windows 登录启动项。电脑或应用关闭时不能实时投递；下次启动会补交已到期的一次性任务。若希望登录后自动恢复，应由 Launcher 注册一个 Windows Task Scheduler 登录启动项，不要为每条提醒各建一个系统任务。

通知适配器使用系统自带 Windows Runtime `Windows.UI.Notifications`，不新增 Python 包。通知标题和正文先 Base64 传给 PowerShell，再进行 XML 转义，不能作为 PowerShell 或 Toast XML 执行。

## 与外部日历的边界

本组不创建 Outlook、QQ 或 Google 日历事件，不邀请参会人，也不定时发邮件。邮箱与日历保持独立工具组；Agent Event 只是延后提交一个正常任务，不绕过任何工具侧限制。
