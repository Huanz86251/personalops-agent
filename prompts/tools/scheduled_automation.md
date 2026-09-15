用 get_current_time 把“明天、下周一、两小时后”等相对时间换成带时区偏移的 ISO-8601 绝对时间。不要让工具猜日期，也不要提交无时区时间。

根据用户明确要求选择投递方式：schedule_create 是 Windows 到期通知；schedule_create_feishu_reminder 把文本直接发回创建它的当前飞书会话，不调用模型；schedule_create_agent_task 到期后把完整任务作为一个新的 SYSTEM/QUEUE Event 持久化并交给 Agent。不得把“提醒我”擅自升级成需要模型和工具执行的 Agent 任务。三个创建工具返回 schedule_id 才表示已经持久化；windows_notify 只立即通知，不会创建未来提醒。

飞书会话地址与 Conversation 必须由宿主从当前可信 Event 绑定，工具参数不允许填写 chat_id。定时 Agent 任务当前只允许 ONCE，并且只能 QUEUE：它不会 INSERT、REPLACE 或打断到期时正在运行的任务。其 payload 是新的完整用户 query，不是修改旧 query，也不是在日程中嵌套另一个 schedule。

暂停、恢复或取消时必须使用 schedule_list 返回的真实 schedule_id，不能根据标题编造。schedule_delete 是保留审计记录的软删除，只能在用户明确要求取消时使用。schedule_runs 中 SUBMITTED 只表示对应投递已提交：不保证用户看到，也不表示排队的 Agent 任务已经完成。

当前能力不等于 Outlook/QQ/Google 日历，不会邀请参会人，也不会定时发送邮件。Agent Event 到期后仍受当时工具权限、确认规则和安全策略约束；创建日程不等于提前授权未来的高风险动作。
