# Scope intent router

这是一个高召回二分类路由器：

- DIRECT_RESPONSE (0)：仅依赖用户当前给出的文字即可回答，不需要读取或改变外部状态。
- REQUIRES_SCOPE_CONTRACT (1)：需要工具、文件、账户、网络、数据库、代码执行或任何外部读写；先交给 Scope Resolver 明确对象、条件和操作。

数据生成时由 Harness 固定标签，Qwen3.7-Flash 只生成同一意图族的表达，不能自行决定标签。当前数据集有 80 个种子族、20,000 条，两个标签各 10,000 条。训练、验证和测试按 seed_id 隔离，避免同源改写泄漏。

主要文件：

- seeds.json：80 个人工种子及边界说明。
- scope_intent_4000.jsonl：首版 4,000 条数据。
- scope_intent_20000.jsonl：扩展后的 20,000 条数据。
- generation_manifest_20000.json：并发生成、用量和公开价估算。
- metrics_minilm_baseline.json：冻结 MiniLM + 逻辑回归基线。
- metrics.json：DistilBERT 全参数微调结果。

正式模型使用 Hugging Face 团队开发的 distilbert/distilbert-base-multilingual-cased，通过 AutoModelForSequenceClassification 微调整个编码器和二分类头。模型权重由 .gitignore 排除，保存在 .models/scope_intent_distilmbert。

运行：

    C:/ProgramData/anaconda3/envs/agent/python.exe scripts/expand_scope_intent_dataset.py
    C:/ProgramData/anaconda3/envs/agent/python.exe scripts/train_scope_intent_classifier.py --epochs 2

当前指标只反映未见过的合成种子族，不等于真实生产准确率。接入 Harness 前应由人工抽查复杂边界，并用真实请求做盲测。
