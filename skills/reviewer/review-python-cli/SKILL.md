---
name: review-python-cli
description: 独立验收 Python 模块、脚本及 CLI，检验真实调用签名、参数、stdout/stderr、退出码与边界；测试失败先排查收集、导入、环境和 Reviewer 测试假设。
metadata:
  roles: "reviewer"
  topics: "verification python cli"
  required-tools: "read_file execute"
---

# Python / CLI 独立验收

## 从契约而非实现推导断言

读取 code_task 的公开签名/命令、MUST、输入输出与错误约定，将每项要求映射到可观察结果。先准备能人工核算的正常样本和最危险的边界，再看候选调用入口。Worker 自测只作线索；不能复制候选算法作为预期答案，也不能只复跑 Worker 自带测试就宣布独立覆盖充分。

## 执行最小有效检查

模块使用真实导入与函数调用，CLI 用子进程验证参数解析、输出流和退出码。按任务选择空输入、中文参数、非法值、依赖失败或重复调用；无关分支不穷举。外部网络/时钟可以替代，但核心逻辑真实执行。检查 JSON 输出的解析及值，不能只查 stdout 含有关键词。

## 测试失败的诊断顺序

本项目 Docker 实测：Reviewer 从 /review 运行，候选在 /workspace；直接 import 候选可能在收集阶段 ModuleNotFoundError。先确认代码确实存在，再按实际包布局设置导入路径。平铺模块可用 `PYTHONPATH=/workspace python -m pytest /review/test_contract.py --collect-only -q -p no:cacheprovider`；src 布局应使用其真实源目录，不能照抄平铺路径。测试和输出留 /review，临时目录按本轮新建，避免 pytest 向只读 /workspace 写缓存或覆盖旧证据。

1. 确认当前候选、工作目录、解释器及项目测试配置；核对实际收集的用例数、退出码、跳过和超时。零用例或全跳过不是通过。
2. 导入/收集失败时先核对包布局；必要时用项目解释器的 `python -m pytest --collect-only -q` 定位，再单跑目标用例。不要随意改路径掩盖候选未交付文件。
3. 断言失败时检查测试是否遵循签名、同步/异步方式、fixture 和真实错误语义。测试假设错就修测试；环境缺失则报告阻塞范围；契约可复现违背才判产品缺陷。
4. 修正测试后重跑相关用例；不可把产品缺陷改成更弱断言、吞异常或无条件 skip。

## 审核输出

为已复现缺陷给 requirement ID、输入/命令、预期与实际、定位及可复跑检查；使用 Reviewer 可写测试位置，不修改产品候选。返修后核对新版本并重跑失败项和相关正常行为。明确区分通过、失败、未验证；收集失败不能包装为业务测试失败或成功。

实测闭环：/review中直接import失败、零收集 → 确认源码后修导入路径，收集12项 → 等于阈值的两项断言失败，其他10项通过 → Worker修复候选，同一12项通过。前半段修验证入口，后半段修产品；不能把两种失败都交给Worker“重写代码”。
