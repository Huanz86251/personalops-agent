# Memory write gate classifier

该目录保存长期记忆 SAVE/SKIP 分类器的可复现实验材料。正式候选以 `MemTensor/MemOperator-0.6B` 为基座，新增 Qwen3 sequence-classification head，并通过 PEFT LoRA 训练；现有多语言 DistilBERT 只作为后续可选基线。

数据由200个固定种子族生成，SAVE与SKIP各100族。Qwen3.7-Flash关闭思考，只负责在固定标签内改写表达，不能决定标签。训练、验证、测试按照seed_id隔离，避免同源改写泄漏。首批100族生成10,000条短中型消息；新增100族生成10,000条目标长度50至250字符的长消息，重点覆盖同一消息混合临时请求与长期事实。数据覆盖中英文、中文名与英文名、亲友和同事关系、邮件、项目、健康、出行、未来任务、一次性问答、实时状态、猜测、引用、删除请求和秘密值。

生成一万条数据：

    C:/ProgramData/anaconda3/envs/agent/python.exe scripts/generate_memory_write_gate_dataset.py
    C:/ProgramData/anaconda3/envs/agent/python.exe scripts/expand_memory_write_gate_long_dataset.py

LoRA训练：

    C:/ProgramData/anaconda3/envs/agent/python.exe scripts/train_memory_write_gate_memoperator.py

合成测试指标不能代替真实用户消息盲测。接入生产门控前，应建立独立人工标注集，并同时检查SAVE召回率、SAVE精确率、临时信息误存率、推理延迟和内存占用。
