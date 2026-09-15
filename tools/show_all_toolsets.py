from langchain.tools import tool


SHOW_ALL_TOOLSETS_NAME = "show_all_toolsets"
SHOW_ALL_TOOLSETS_PREFIX = "SHOW_ALL_TOOLSETS:"
SHOW_ALL_TOOLSETS_REASON_MAX_CHARS = 400


def show_all_toolsets(reason: str) -> str:
    """当前可见工具都不适用时，显示本Worker可用的全部工具。

    这是最后的工具发现入口。当前可见工具可以完成任务时不要调用；
    不要为了重试业务错误、登录失败或参数错误而调用。
    调用成功后，下一轮会一次性显示本Worker真实注册的全部工具，
    包括每个工具的说明和完整参数Schema。

    Args:
        reason: 先说明当前可见工具为什么都不能完成下一步；
            不要填写猜测的工具名或工具组名。

    Returns:
        供Harness识别的全部工具展示请求。
    """
    normalized_reason = " ".join(reason.strip().split())
    if not normalized_reason:
        return "显示全部工具失败：reason不能为空。"
    normalized_reason = normalized_reason[:SHOW_ALL_TOOLSETS_REASON_MAX_CHARS]
    return f"{SHOW_ALL_TOOLSETS_PREFIX} {normalized_reason}"


show_all_toolsets_tool = tool(parse_docstring=True)(show_all_toolsets)
