# LOCAL_ANALYSIS 能力卡

## 选择它

任务明确需要符号数学、代数化简、方程求解，或只做 Python 语法检查、Ruff 静态检查而不执行代码。

## 不选择它

- 需要运行程序、pytest、安装依赖、Git 或 Shell 调试。
- 只要求解释数学概念，不要求计算。
- 数据分析、训练模型或执行任意 Python 脚本。

## 典型命令

- 用符号计算解这个方程并化简结果。
- 检查这段 Python 有没有语法错误。
- 对这个文件运行 Ruff 静态检查，不执行代码。

## 易混淆边界

SOFTWARE_DEVELOPMENT 负责可执行的开发和测试闭环；LOCAL_ANALYSIS 只提供受限的符号计算与非执行式代码检查。需要先读源码时可同时选择 FILE_INSPECTION。
