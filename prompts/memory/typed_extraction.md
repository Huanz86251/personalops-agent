第二轮只填写上一轮已经列出的 candidate_id、frame_id 和 frame_type，不得新增 frame。原文和 source_time 已在对话前缀中，不要复述原话，不要输出 evidence；宿主会按 candidate_id 自动回填原始证据、来源与记录时间。每个 frame 输出一条独立 record，并写一条简洁、可独立召回的中文 summary，例如“阿哈默德是用户的导师”。

importance 表示长期使用价值：low、medium、high、urgent；urgent 仅用于延误会造成近期实质后果的事项。confidence 表示原文支持强度：明确陈述用 high，合理但不完全明确用 medium，含糊或推测用 low。二者不能混用。

日期必须是带时区的 ISO 8601；无法可靠确定就用 null。profile 的 name 只用于用户明确陈述自己的姓名；“以后叫我小黄”这类希望助手如何称呼用户的要求使用 preferred_name，不使用 name、other 或 preference。人物关系中的 relation 是该人物相对于用户的角色；other 必须填写 other_relation。task 的 request 包含普通请求或委托，order 只用于明确命令；promise 是某人明确承诺，reminder 是提醒，update 是任务状态或期限变化，cancel 是取消。action 保留简短常用动作，event 不得照抄自然语言动词创造新类型。

只输出 JSON：
{{schema}}
