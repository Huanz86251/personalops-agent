from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

from pathlib import Path
from urllib.error import (
    URLError,
)
from urllib.request import (
    ProxyHandler,
    Request,
    build_opener,
)

from path import (
    AGENT_DATA_ROOT,
)


logger = logging.getLogger(
    "agent"
)


PHOENIX_HOST = (
    os.getenv(
        "PHOENIX_HOST",
        "127.0.0.1",
    ).strip()
    or "127.0.0.1"
)

try:
    PHOENIX_PORT = int(
        os.getenv(
            "PHOENIX_PORT",
            "6007",
        ).strip()
    )

except ValueError as error:
    raise RuntimeError(
        "PHOENIX_PORT必须是整数。"
    ) from error


try:
    PHOENIX_STARTUP_TIMEOUT_SECONDS = float(
        os.getenv(
            "PHOENIX_STARTUP_TIMEOUT_SECONDS",
            "120",
        ).strip()
    )

except ValueError as error:
    raise RuntimeError(
        "PHOENIX_STARTUP_TIMEOUT_SECONDS"
        "必须是数字。"
    ) from error


class PhoenixServerRuntime:
    """管理本地Phoenix Server子进程。

    主要职责：

    1. 启动前检查Phoenix是否已经运行；
    2. 未运行时自动启动本地Phoenix；
    3. 等待数据库和HTTP服务就绪；
    4. 固定Phoenix持久化目录；
    5. 程序关闭时只关闭自己创建的进程。

    如果6006端口上的Phoenix是用户手动启动的，
    当前Runtime只会复用，不会擅自关闭。
    """

    def __init__(
        self,
        host: str = PHOENIX_HOST,
        port: int = PHOENIX_PORT,
        startup_timeout_seconds: float = (
            PHOENIX_STARTUP_TIMEOUT_SECONDS
        ),
    ) -> None:
        self.host = host
        self.port = port

        self.startup_timeout_seconds = (
            startup_timeout_seconds
        )

        self.working_directory = (
            AGENT_DATA_ROOT
            / "phoenix"
        )

        self.database_path = (
            self.working_directory
            / "phoenix.db"
        )

        self.log_path = (
            self.working_directory
            / "phoenix-server.log"
        )

        self._process: (
            subprocess.Popen
            | None
        ) = None

        self._log_stream = None

        # True表示Phoenix是当前Agent启动的，
        # 退出时才有权关闭。
        self._owns_process = False


    @property
    def ui_url(
        self,
    ) -> str:
        """返回Phoenix网页地址。"""

        return (
            f"http://{self.host}:"
            f"{self.port}"
        )

    @property
    def ready_url(
        self,
    ) -> str:
        """返回Phoenix数据库就绪检查地址。"""

        return (
            f"{self.ui_url}/readyz"
        )

    @property
    def is_owned(
        self,
    ) -> bool:
        """判断当前Phoenix是否由本Runtime启动。"""

        return (
            self._owns_process
        )

    @staticmethod
    def _append_no_proxy(
        environment: dict[
            str,
            str,
        ],
    ) -> None:
        """只为Phoenix子进程添加本机代理绕过。

        不修改当前Agent进程的环境，
        因此不会影响Telegram或DeepSeek。
        """

        local_addresses = (
            "127.0.0.1",
            "localhost",
            "::1",
        )

        for variable_name in (
            "NO_PROXY",
            "no_proxy",
        ):
            current_items = [
                item.strip()

                for item in environment.get(
                    variable_name,
                    "",
                ).split(",")

                if item.strip()
            ]

            existing_items = {
                item.casefold()

                for item in current_items
            }

            for address in local_addresses:
                if (
                    address.casefold()
                    not in existing_items
                ):
                    current_items.append(
                        address
                    )

            environment[
                variable_name
            ] = ",".join(
                current_items
            )

    def _build_environment(
        self,
    ) -> dict[
        str,
        str,
    ]:
        """生成Phoenix子进程使用的环境变量。"""

        environment = (
            os.environ.copy()
        )

        environment[
            "PHOENIX_HOST"
        ] = self.host

        environment[
            "PHOENIX_PORT"
        ] = str(
            self.port
        )

        environment[
            "PHOENIX_WORKING_DIR"
        ] = str(
            self.working_directory
        )

        # Windows绝对路径需要转换成：
        #
        # sqlite:///D:/PythonProject/...
        database_url = (
            "sqlite:///"
            f"{self.database_path.as_posix()}"
        )

        environment[
            "PHOENIX_SQL_DATABASE_URL"
        ] = database_url

        self._append_no_proxy(
            environment
        )

        return environment

    def _is_ready(
        self,
    ) -> bool:
        """直接检查Phoenix数据库和HTTP服务是否就绪。

        ProxyHandler({})表示本次健康检查
        明确不使用系统HTTP代理。
        """

        opener = build_opener(
            ProxyHandler(
                {}
            )
        )

        request = Request(
            self.ready_url,

            method="GET",

            headers={
                "User-Agent": (
                    "agentnew-phoenix-runtime"
                ),
            },
        )

        try:
            with opener.open(
                request,

                timeout=1.0,
            ) as response:
                return (
                    response.status
                    == 200
                )

        except (
            URLError,
            TimeoutError,
            OSError,
        ):
            return False

    def _read_log_tail(
        self,
        max_chars: int = 4000,
    ) -> str:
        """读取Phoenix日志末尾，便于启动失败排查。"""

        if not self.log_path.is_file():
            return ""

        try:
            content = (
                self.log_path
                .read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

        except OSError:
            return ""

        return content[
            -max_chars:
        ]

    def start(
        self,
    ) -> bool:
        """确保本地Phoenix Server已经启动。

        Returns:
            True：
                当前Runtime创建了Phoenix子进程。

            False：
                Phoenix本来就已运行，直接复用。
        """

        if self._is_ready():
            logger.info(
                "Phoenix本地服务已存在，"
                "本次直接复用 | "
                "ui=%s",

                self.ui_url,
            )

            self._owns_process = False

            return False

        self.working_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._log_stream = (
            self.log_path.open(
                mode="a",
                encoding="utf-8",
            )
        )

        command = [
            sys.executable,

            "-m",
            "phoenix.server.main",

            "serve",

            "--host",
            self.host,

            "--port",
            str(
                self.port
            ),
        ]

        creation_flags = 0

        if os.name == "nt":
            creation_flags = (
                subprocess
                .CREATE_NEW_PROCESS_GROUP
            )

        process = subprocess.Popen(
            command,

            cwd=str(
                AGENT_DATA_ROOT.parent
            ),

            env=(
                self._build_environment()
            ),

            stdout=(
                self._log_stream
            ),

            stderr=(
                subprocess.STDOUT
            ),

            creationflags=(
                creation_flags
            ),
        )

        self._process = process
        self._owns_process = True

        deadline = (
            time.monotonic()
            + self.startup_timeout_seconds
        )

        while (
            time.monotonic()
            < deadline
        ):
            if self._is_ready():
                logger.info(
                    "Phoenix本地服务已自动启动 | "
                    "ui=%s | database=%s",

                    self.ui_url,

                    self.database_path,
                )

                return True

            exit_code = (
                process.poll()
            )

            if exit_code is not None:
                log_tail = (
                    self._read_log_tail()
                )

                self._close_log_stream()

                self._process = None
                self._owns_process = False

                raise RuntimeError(
                    "Phoenix Server启动后异常退出 | "
                    f"exit_code={exit_code}\n"
                    f"日志：{self.log_path}\n\n"
                    f"{log_tail}"
                )

            time.sleep(
                0.25
            )

        log_tail = (
            self._read_log_tail()
        )

        self.stop()

        raise TimeoutError(
            "等待Phoenix Server启动超时 | "
            f"url={self.ready_url}\n"
            f"日志：{self.log_path}\n\n"
            f"{log_tail}"
        )

    def _close_log_stream(
        self,
    ) -> None:
        """关闭Phoenix日志文件句柄。"""

        log_stream = (
            self._log_stream
        )

        self._log_stream = None

        if log_stream is None:
            return

        try:
            log_stream.close()

        except OSError:
            pass

    def stop(
        self,
    ) -> None:
        """关闭当前Runtime创建的Phoenix子进程。"""

        process = (
            self._process
        )

        # 不是我们启动的Phoenix，
        # 绝对不能关闭。
        if (
            process is None
            or not self._owns_process
        ):
            self._close_log_stream()
            return

        try:
            if process.poll() is None:
                try:
                    if os.name == "nt":
                        process.send_signal(
                            signal.CTRL_BREAK_EVENT
                        )

                    else:
                        process.terminate()

                    process.wait(
                        timeout=8.0
                    )

                except (
                    OSError,
                    ValueError,
                    subprocess.TimeoutExpired,
                ):
                    process.terminate()

                    try:
                        process.wait(
                            timeout=3.0
                        )

                    except subprocess.TimeoutExpired:
                        process.kill()

                        process.wait(
                            timeout=3.0
                        )

        finally:
            self._process = None
            self._owns_process = False

            self._close_log_stream()

        logger.info(
            "Phoenix本地服务已关闭"
        )