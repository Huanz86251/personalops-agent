---
name: general-create-reminders
description: 创建一次、每日或每周的 Windows/飞书提醒、立即通知或到期 Agent 任务时，解析带时区绝对时间并选择正确投递方式；不管理已有提醒或创建外部日历会议。
metadata:
  roles: "general"
  topics: "reminders schedules"
  required-tools: "get_current_time schedule_create"
  required-toolsets: "SCHEDULE_CREATION"
  exclusive: "false"
---

# 创建本地提醒或到期任务

## 确定时间和投递方式

相对时间先用 get_current_time，再换算为带时区的绝对 run_at；明确标题、提醒内容、timezone_name 与 recurrence。当前只支持 ONCE/DAILY/WEEKLY，不能把“每月”“工作日”“五分钟一次”静默替换成每日。时间歧义影响结果时先询问，用户已给明确信息时不重复确认。

普通“提醒我”默认创建 Windows 通知；明确说“在飞书提醒”才选择飞书文本；只有“到时帮我查询/整理/处理”才创建一次性 Agent 任务。不得把低成本提醒擅自升级成调用模型与工具的 Agent 任务。

## 创建并核对真实结果

按用户要求调用 schedule_create、schedule_create_feishu_reminder 或 schedule_create_agent_task；立即通知才用 windows_notify。只有工具返回 schedule_id 才说明已持久化。创建结果未知时如有管理能力先查询可能已创建项，避免重复；没有管理能力则诚实报告未知，不盲目重试。

飞书会话由宿主绑定，不填写或猜测 chat_id。Agent 到期任务只能 ONCE/QUEUE，不会打断当前任务，也不代表提前批准未来高风险操作。

## 自检与回报

报告真实 schedule_id、绝对时间、时区、重复规则、投递方式和当前状态。Windows 或飞书返回 SUBMITTED 只表示投递已提交，不证明用户看到。服务离线、创建失败或时间不受支持时明确未设置成功，不用临时后台脚本绕过。
