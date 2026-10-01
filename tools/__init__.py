from .time_tools import (
    get_current_time,
    get_current_time_tool,
)
from .web_tools import (
    find_github_mirror,
    github_mirror_tool,
    web_search,
    web_search_tool,
)
from .local_native import LOCAL_NATIVE_TOOLS
from .feishu_file_tools import send_local_file_to_feishu
from .schedule_tools import SCHEDULE_TOOLS
from .desktop_tools import DESKTOP_TOOLS
from .host_file_tools import HOST_FILE_TOOLS
from .market_1688 import market_1688_is_configured, search_1688_products_tool
from .cinema_showtimes import (
    list_cinemas_tool, list_city_movies_tool,
    get_cinema_showtimes_tool, get_movie_showtimes_tool,
)

ALL_TOOLS = [
    get_current_time_tool,

    web_search_tool,
    github_mirror_tool,
    list_cinemas_tool,
    list_city_movies_tool,
    get_cinema_showtimes_tool,
    get_movie_showtimes_tool,

    *LOCAL_NATIVE_TOOLS,
    *SCHEDULE_TOOLS,
    *DESKTOP_TOOLS,
    *HOST_FILE_TOOLS,
    send_local_file_to_feishu,
]

if market_1688_is_configured():
    ALL_TOOLS.append(search_1688_products_tool)


__all__ = [
    "ALL_TOOLS",

    "get_current_time",
    "get_current_time_tool",

    "web_search",
    "web_search_tool",

    "find_github_mirror",
    "github_mirror_tool",

]
