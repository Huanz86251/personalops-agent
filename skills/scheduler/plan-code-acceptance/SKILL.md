---
name: plan-code-acceptance
description: 规划代码、页面、文件转换或 Schema 任务时，生成让 Worker 和独立 Reviewer 都能执行的验收契约，明确公开入口、输入输出、错误语义与所需证据，而不是只写“实现并测试”。
metadata:
  roles: "scheduler"
  topics: "planning implementation contracts"
---

# 可独立验收的任务契约

## 先确定结果与能力

从用户请求抽取必须实现的外部行为、交付形式与限制，再依据真实工具目录确定由谁执行。不要把“本地 agent 能做”误等同于 Code 沙箱能操作宿主；Code 默认断网且没有保证浏览器，邮件、提醒等业务操作走实际具备工具的 General。

## 写进现有结构，不创造协议

- requirements：每项 MUST 描述输入/前置状态、操作及可观察结果；SHOULD 单列偏好。不要把某种实现手法和额外美化升级为硬要求。
- interfaces：用真实 kind 与 details 写清公开入口和参数/格式。未知接口先安排读取或探索，不编造仓库路径与函数。
- validation_expectations：要求独立检查哪些行为及证据层级，不给 Reviewer 一段必须照抄的测试代码。检查设计应允许揭示实现错误。
- implementation_guidance 与 non_goals：承载建议、环境限制和明确不做的事情，不能替代 MUST。

## 根据任务选择检查

页面：入口、关键交互的初始/成功/空/错误状态；真实浏览器不可用时声明渲染证据缺口。Python/CLI：签名或参数、返回值/输出流、失败语义。文件：格式、字段、筛选、覆盖约定及执行后重读。Schema：缺失/null、类型、额外字段和跨字段约束。只选相关分支，简单任务不硬拆多步。

CSV实测契约明确“金额>=阈值、零合法、非法配置退出2、不覆盖已有文件”，Reviewer无需Worker对话就能设计等值/零/坏输入检查，并发现仅正常样本漏掉的比较符缺陷。契约应提供可判真假的行为，而非指定一段测试答案；未规定的非法记录策略按现有规范或澄清处理，不擅自允许丢数据。

## 交接与复查

具体写出容器可见输入、源码/命令入口和输出位置：/handoff 只读输入，Worker 的 /workspace 候选由 Reviewer 只读访问，Reviewer 测试输出在 /review。不要把“文件已写出”当作其他角色能读到，更不要给宿主路径让容器执行。明确依赖是否可用；本机镜像只有被实际检查过的能力，宿主装了 Pydantic 不代表容器也有。

Step 的 success_criteria 与 code_task 保持一致，skill_topics 用最多三个业务/技术主题辅助候选，不强迫所有角色选相同 Skill。提交前问：Reviewer 不看 Worker 对话，能否知道调用什么、什么算正确、哪些仍无法验证？若答案是否，就补缺失契约，不再增加口号。
