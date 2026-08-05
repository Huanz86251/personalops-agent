from __future__ import annotations

import logging
import shutil
from contextlib import (
    AsyncExitStack,
)
from pathlib import Path

from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
)
from langchain_mcp_adapters.tools import (
    load_mcp_tools,
)

from path import AGENT_DATA_ROOT


logger = logging.getLogger(
    "agent"
)


# 为了面试演示和环境可复现，
# 不在每次启动时使用不确定的latest版本。
#
# 截至2026-07-23，
# 当前稳定版本是0.0.78。
PLAYWRIGHT_MCP_PACKAGE = (
    "@playwright/mcp@0.0.78"
)


PLAYWRIGHT_PROFILE_PATH = (
    AGENT_DATA_ROOT
    / "playwright-profile"
)

PLAYWRIGHT_OUTPUT_PATH = (
    AGENT_DATA_ROOT
    / "playwright-output"
)


# MCP Server可能暴露很多工具。
#
# 我们不把所有工具无条件交给模型，
# 而是只允许当前Agent真正需要的浏览器能力。
#
# 特别是不注册：
# browser_run_code_unsafe
# browser_evaluate
# browser_file_upload
# browser_cookie_set
# browser_localstorage_set
PLAYWRIGHT_ALLOWED_TOOLS = {
    "browser_navigate",
    "browser_navigate_back",

    "browser_snapshot",
    "browser_find",

    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_wait_for",

    "browser_tabs",
    "browser_close",
}


PLAYWRIGHT_REQUIRED_TOOLS = {
    "browser_navigate",
    "browser_snapshot",
    "browser_close",
}


class PlaywrightMCPRuntime:
    """管理Playwright MCP Server和持久ClientSession。

    整个Agent程序只启动一个Playwright MCP子进程。

    MCP Server负责：
    - 浏览器实例
    - 页面状态
    - 标签页
    - Cookie和登录状态
    - 网页结构快照
    - 页面交互

    当前Python程序作为MCP Client，
    负责启动Server、加载工具和管理生命周期。
    """

    def __init__(
        self,
        profile_path: Path = (
            PLAYWRIGHT_PROFILE_PATH
        ),
        output_path: Path = (
            PLAYWRIGHT_OUTPUT_PATH
        ),
    ) -> None:
        self.profile_path = (
            profile_path
        )

        self.output_path = (
            output_path
        )

        self._client: (
            MultiServerMCPClient
            | None
        ) = None

        self._session = None

        self._exit_stack: (
            AsyncExitStack
            | None
        ) = None

        self._tools: list = []

        self._tools_by_name: dict[
            str,
            object,
        ] = {}

    @property
    def tools(
        self,
    ) -> list:
        """返回可注册到Agent的MCP工具。"""

        return list(
            self._tools
        )

    @property
    def is_started(
        self,
    ) -> bool:
        """判断MCP持久Session是否已经启动。"""

        return (
            self._exit_stack
            is not None
        )

    @staticmethod
    def _find_npx_command() -> str:
        """寻找Windows或其他系统中的npx命令。"""

        npx_command = (
            shutil.which(
                "npx.cmd"
            )
            or shutil.which(
                "npx"
            )
        )

        if npx_command is None:
            raise RuntimeError(
                "没有找到npx命令。"
                "请先安装Node.js 20或更高版本，"
                "并确认npx --version可以正常运行。"
            )

        return npx_command

    def _build_server_config(
        self,
    ) -> dict:
        """生成Playwright MCP stdio配置。"""

        npx_command = (
            self._find_npx_command()
        )

        return {
            "playwright": {
                "transport": "stdio",

                "command": (
                    npx_command
                ),

                "args": [
                    "-y",

                    PLAYWRIGHT_MCP_PACKAGE,

                    # Windows默认自带Edge，
                    # 不依赖用户安装Chrome。
                    "--browser=msedge",

                    # 默认保持有界面模式，
                    # 演示时面试官可以看到浏览器操作。
                    (
                        "--user-data-dir="
                        f"{self.profile_path}"
                    ),

                    (
                        "--output-dir="
                        f"{self.output_path}"
                    ),

                    "--viewport-size=1280x900",

                    "--timeout-action=10000",

                    "--timeout-navigation=60000",
                ],
            }
        }

    async def start(
        self,
    ) -> None:
        """启动MCP Server并建立持久ClientSession。"""

        if self.is_started:
            return

        self.profile_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        exit_stack = (
            AsyncExitStack()
        )

        client = (
            MultiServerMCPClient(
                self._build_server_config()
            )
        )

        try:
            # client.session()进入时：
            #
            # 1. 自动启动npx子进程；
            # 2. 与Playwright MCP建立stdio连接；
            # 3. 初始化MCP ClientSession。
            #
            # AsyncExitStack会在stop()时
            # 自动退出Session并回收子进程。
            session = await (
                exit_stack
                .enter_async_context(
                    client.session(
                        "playwright"
                    )
                )
            )

            all_tools = await (
                load_mcp_tools(
                    session
                )
            )

            allowed_tools = [
                current_tool

                for current_tool
                in all_tools

                if current_tool.name
                in PLAYWRIGHT_ALLOWED_TOOLS
            ]

            loaded_names = {
                current_tool.name

                for current_tool
                in allowed_tools
            }

            missing_names = (
                PLAYWRIGHT_REQUIRED_TOOLS
                - loaded_names
            )

            if missing_names:
                raise RuntimeError(
                    "Playwright MCP缺少必要工具："
                    + ", ".join(
                        sorted(
                            missing_names
                        )
                    )
                )

        except Exception:
            await exit_stack.aclose()
            raise

        self._client = client
        self._session = session
        self._exit_stack = (
            exit_stack
        )

        self._tools = (
            allowed_tools
        )

        self._tools_by_name = {
            current_tool.name: (
                current_tool
            )

            for current_tool
            in allowed_tools
        }

        excluded_names = sorted(
            current_tool.name

            for current_tool
            in all_tools

            if current_tool.name
            not in loaded_names
        )

        logger.info(
            "Playwright MCP已启动 | "
            "transport=stdio | "
            "browser=msedge | "
            "allowed_tools=%s",

            sorted(
                loaded_names
            ),
        )

        if excluded_names:
            logger.info(
                "Playwright MCP工具已按白名单过滤 | "
                "excluded_tools=%s",

                excluded_names,
            )

    async def reset_page(
        self,
    ) -> None:
        """关闭当前浏览器页面，隔离不同Conversation。

        关闭页面不会删除独立Profile，
        所以Cookie和用户手动建立的登录状态仍会保留。
        """

        close_tool = (
            self._tools_by_name.get(
                "browser_close"
            )
        )

        if close_tool is None:
            return

        try:
            await close_tool.ainvoke(
                {}
            )

        except Exception:
            # 页面可能本来就没有打开。
            # 这种错误不应影响创建或切换Conversation。
            logger.debug(
                "关闭Playwright当前页面失败，"
                "可能当前没有活动页面。",
                exc_info=True,
            )

    async def stop(
        self,
    ) -> None:
        """关闭MCP Session并回收子进程。"""

        exit_stack = (
            self._exit_stack
        )

        if exit_stack is None:
            return

        try:
            await self.reset_page()

        finally:
            self._tools = []
            self._tools_by_name.clear()

            self._session = None
            self._client = None
            self._exit_stack = None

            await exit_stack.aclose()

        logger.info(
            "Playwright MCP已正常关闭。"
        )