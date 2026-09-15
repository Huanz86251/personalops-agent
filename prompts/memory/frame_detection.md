<!-- include: memory/extraction_base -->

第一轮只做分帧，不做详细抽取。逐个处理 candidate_id；每条输入可有 0–5 个独立 frame。没有长期价值就返回空 frames。frame_type 只用：profile（用户资料）、preference（喜欢/不喜欢/偏好/避免）、person_relation（人与用户的常见关系）、project（项目事实）、task（请求、命令、承诺、提醒、进度更新、取消）。不保存普通闲聊、一次性操作指令或未确认推测。不复述原话，不输出 evidence、摘要、人物名称、时间、重要性或置信度。frame_id 仅需在当前 candidate 内唯一，如 f1、f2。

例：输入“阿哈默德是我的导师，他让我三周内写完文档”可返回 person_relation 与 task 两个 frame；“以后叫我小黄”返回 profile；输入“把这句话翻译成中文”返回空 frames。

输出格式（JSON Schema）：
{{schema}}
