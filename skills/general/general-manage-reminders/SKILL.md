---
name: general-manage-reminders
description: 创建、查询或管理 Windows 桌面一次/每日/每周提醒时，解析带时区的绝对时间、复用真实 schedule ID 并验证状态；区分计划创建、系统通知受理与用户已看到。
metadata:
  roles: "general"
  topics: "reminders schedules"
  required-tools: "get_current_time schedule_create schedule_list"
---

# 本地提醒任务

## 确定时间与请求类型

相对时间先用 get_current_time，再换算带时区的绝对 run_at；明确标题、提醒内容、timezone_name 与 recurrence。当前仅 ONCE/DAILY/WEEKLY，不能把“每月”“工作日”“五分钟一次”悄悄替换成每日。日期/上午下午/时区不明且影响结果时先问；用户已给明确时间不重复确认。

## 使用真实工具管理

新建用 schedule_create；查询或修改已有任务先 schedule_list，以真实 schedule_id 选择对象，重名需结合时间和内容。暂停、恢复、删除、运行历史仅在相应工具实际存在时调用。当前没有 schedule_update；调整时间不能谎称原位更新，也不能未经授权顺带取消其他提醒。

创建后检查真实返回并按需通过 schedule_list 核对时间、时区、重复规则及状态。创建结果未知时先查询可能已创建的项，不立即重复创建。需要变更为新计划时说明当前工具只能新建与取消，在用户变更请求范围内确认目标、核对新项、处理旧项，并报告部分成功，防止重复提醒。

## 解释运行边界

服务随 PersonalOps 运行；离线期间不保证按时弹出，也没有保证 Windows 开机自启。重启补交与重复任务跳过积压按现有服务语义，不承诺严格实时或恰好一次。没有运行服务或创建失败时明确未设置成功，不转用临时后台脚本绕过。

windows_notify 若可用只用于用户要求的即时通知，不等同于持久定时任务。schedule_runs 中 SUBMITTED 仅证明 Windows 接受通知，不证明用户看见或已完成待办。

## 自检与回报

General直接报告真实ID、时间/时区、重复规则、状态与缺口。临时存储测试中无时区run_at被拒，补+08:00后正确换算并持久化返回ID；虚拟通知器的SUBMITTED只证明受理口径，不证明真实桌面已弹出。遇时间校验错修字段，不重建整个任务。只查历史时按需用status=ALL，默认ACTIVE漏掉已完成/取消项不能说明记录不存在；查询不授权新建或恢复。
