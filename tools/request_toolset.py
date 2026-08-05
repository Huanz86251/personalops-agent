from langchain.tools import (
    tool,
)


REQUEST_TOOLSET_NAME = (
    "request_toolset"
)


TOOLSET_REQUEST_PREFIX = (
    "TOOLSET_REQUEST:"
)


REQUEST_TOOLSET_TASK_MAX_CHARS = 400


def request_toolset(
    task: str,
) -> str:
    """当前可见工具不足以完成下一步时，请求重新选择工具组。

    只描述下一步需要完成的具体任务，
    不要填写工具名称或工具组名称。
    当前工具已经足够时不要调用。

    Args:
        task: 下一步需要完成的具体任务，
            不要重复已经完成的步骤。

    Returns:
        供工具组Router读取的新请求。
    """

    normalized_task = (
        " ".join(
            task
            .strip()
            .split()
        )
    )

    if not normalized_task:
        return (
            "请求工具组失败："
            "task不能为空。"
        )

    normalized_task = (
        normalized_task[
            :REQUEST_TOOLSET_TASK_MAX_CHARS
        ]
    )

    return (
        f"{TOOLSET_REQUEST_PREFIX} "
        f"{normalized_task}"
    )


request_toolset_tool = tool(
    parse_docstring=True
)(
    request_toolset
)