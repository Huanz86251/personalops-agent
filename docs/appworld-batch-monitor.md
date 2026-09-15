# AppWorld 私有批次监控器

批次监控器在每道 `run_conversation` 完成并写出结果、用量、费用与 Span 后运行。它不调用模型，不进入 Agent 上下文，也不复制任务原文、模型消息、工具返回或凭据。

## 连续运行

为同一批次指定稳定名称：

```text
python -m evals.appworld.run_conversation --task <TRAIN_TASK_ID> --split train --allow-paid --monitor-batch <BATCH_ID>
```

默认挂载监控器。临时卸载可加 `--no-monitor`；全局默认可设 `APPWORLD_RUN_MONITOR_ENABLED=false`。`APPWORLD_MONITOR_BATCH_ID` 可作为默认批次名。

每道题结束后更新私有目录 `.agent/appworld-monitor/<BATCH_ID>/`：

- `report.md`：失败/未评分优先，随后列出费用、Token、耗时和角色 Token 热点；有 Phoenix 数据时提供本地直达链接。
- `index.csv`：适合排序和筛选。
- `index.json`：完整的机器可读统计。
- `runs/<TRIAL_ID>.json`：逐次运行索引，便于中断后恢复。

PASSED/FAILED 只采用官方评测。没有官方结果时标为 UNSCORED；缺失费用、Span 或用量时保持未知。

## 重建历史索引

不重新运行任务、不调用模型，只扫描已有私有记录：

```text
python -m evals.appworld.batch_monitor --batch-id <BATCH_ID> --latest 40
```

也可在命令末尾传入指定运行目录。旧运行没有 `spans.private.json` 时，Phoenix 链接和耗时会显示未知。

## 批次结束后的查看顺序

1. 先看 `report.md` 的“失败与未评分”，只打开这些 Trace。
2. 再看“费用最高”“Token最高”“耗时最长”，定位长循环或重复工作。
3. 最后看“按角色Token热点”，判断成本集中在 Scheduler、Worker、Reporter 还是 Reviewer。

索引仅保存在 Git 忽略的 `.agent/` 中，不可发布原始 AppWorld 数据或 Trace。
