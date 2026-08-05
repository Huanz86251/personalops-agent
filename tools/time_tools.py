from datetime import datetime
from zoneinfo import (
    ZoneInfo,
    ZoneInfoNotFoundError,
)

from langchain.tools import tool


def get_current_time(
    timezone_name: str = "Asia/Shanghai",
) -> str:
    """获取指定时区的当前日期、时间和星期。

    Args:
        timezone_name: IANA时区名称，例如Asia/Shanghai或America/Toronto。
    """

    try:
        now = datetime.now(
            ZoneInfo(
                timezone_name
            )
        )

    except ZoneInfoNotFoundError:
        return (
            "无法识别时区："
            f"{timezone_name}"
        )

    weekday_names = (
        "星期一",
        "星期二",
        "星期三",
        "星期四",
        "星期五",
        "星期六",
        "星期日",
    )

    weekday = weekday_names[
        now.weekday()
    ]

    return (
        f"时区：{timezone_name}\n"
        f"当前时间："
        f"{now.isoformat(timespec='seconds')}\n"
        f"星期：{weekday}"
    )

get_current_time_tool = tool(
    parse_docstring=True
)(get_current_time)