# 面试现场：不用背命令

## 正常演示

1. 在项目左侧打开 interview_demo 文件夹。
2. 打开 01_一键运行零模型评测并打开Phoenix.py。
3. 点击编辑器右上角绿色运行按钮。
4. 等待终端显示“演示已经准备好”，Chrome 会自动打开。
5. 在 Phoenix 中进入 personalops-eval-infrastructure。
6. 打开最新的 EVAL / Infrastructure acceptance。
7. 依次展开 PREPARE、RUN、两个 TOOL / AppWorld execute、GRADE。
8. 演示结束后回到运行窗口按 Enter，脚本只关闭自己启动的 Phoenix。

这条演示真实运行 Docker、AppWorld 工具协议和官方 grader，但外部模型调用为 0。
official_task_success=false 是预期，因为它只验证基础设施，没有尝试解决任务。

## 现场保险方案

如果 Docker 启动慢或面试时间很紧：

1. 打开 02_只打开Phoenix查看已有结果.py。
2. 点击右上角运行。
3. 它直接打开已保存的 Trace，不运行 Docker，不调用模型。

## 不要现场误点的内容

不要为了展示而运行 evals/appworld/run.py。它是真实 Agent 质量评测，会读取配置
并调用外部模型，产生费用。需要展示真实 Agent Trace 时，提前完成受控运行并
在现场使用“02 只查看已有结果”即可。


## 真实 Agent 评测

需要产生一条新的真实模型 Trace 时：

1. 打开 03_运行真实Agent评测_会调用模型.py。
2. 点击右上角运行。
3. 脚本固定只跑一条 Train calibration，最多 24 次模型调用。
4. 完成后 Chrome 自动打开；选择 personalops-eval-private-learning。
5. 打开最新 EVAL / AppWorld trial，查看四个阶段、LLM、TOOL 和官方 GRADE。

这条脚本会产生真实 provider 费用，但不会自动运行下一题或批量任务。
