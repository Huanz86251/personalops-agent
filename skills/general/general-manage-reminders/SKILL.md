---
name: general-manage-reminders
description: 查询、暂停、恢复、取消已有提醒或查看运行记录时，先取得真实 schedule ID 并核对状态；不负责新建提醒或外部日历会议。
metadata:
  roles: "general"
  topics: "reminders schedules"
  required-tools: "schedule_list"
  required-toolsets: "SCHEDULE_MANAGEMENT"
  exclusive: "false"
---

# 已有提醒的查询与管理

## 使用真实工具管理

查询或修改已有任务先 schedule_list，以真实 schedule_id 选择对象，重名需结合时间和内容。暂停、恢复、删除、运行历史仅在相应工具实际存在时调用。当前没有 schedule_update；调整时间不能谎称原位更新，也不能未经授权顺带取消其他提醒。需要创建替代项时必须另有 SCHEDULE_CREATION 能力与用户授权。

操作后按需通过 schedule_list 核对状态。结果未知时先查询目标项，不立即重复暂停、恢复或取消。需要变更为新计划时说明当前管理工具不能原位修改时间，交给创建能力处理新项，并在用户范围内处理旧项。

## 解释运行边界

服务随 PersonalOps 运行；离线期间不保证按时弹出，也没有保证 Windows 开机自启。重启补交与重复任务跳过积压按现有服务语义，不承诺严格实时或恰好一次。没有运行服务或创建失败时明确未设置成功，不转用临时后台脚本绕过。

## 自检与回报

General直接报告真实ID、时间/时区、重复规则、状态与缺口。schedule_runs 中 SUBMITTED 只证明投递受理，不证明真实桌面已弹出或用户完成待办。只查历史时按需用status=ALL，默认ACTIVE漏掉已完成/取消项不能说明记录不存在；查询不授权新建、恢复或取消。
