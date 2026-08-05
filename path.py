from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


# Agent可以操作的工作目录。
#
# 文件工具中的相对路径，
# 以及Shell默认工作目录，
# 都以这里作为起点。
WORKSPACE_ROOT = (
    PROJECT_ROOT
    / "workspace"
)


# Agent程序自己的内部持久化目录。
#
# 这里保存：
# - LangGraph Checkpoint；
# - 跨Conversation长期记忆；
# - Playwright持久Profile；
# - Playwright运行输出。
#
# 这个目录不属于模型工作区，
# 后续文件工具将明确禁止访问它。
AGENT_DATA_ROOT = (
    PROJECT_ROOT
    / ".agent"
)
AGENT_DATA_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

WORKSPACE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)