from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import PureWindowsPath
from typing import Literal
from urllib.parse import urlsplit

from path import WORKSPACE_ROOT


logger = logging.getLogger("agent")


ProgressStage = Literal[
    "PLAN_CREATED",
    "STEP_STARTED",
    "CAPABILITY_SELECTED",
    "STEP_COMPLETED",
    "STEP_PARTIAL",
    "STEP_BLOCKED",
    "STEP_FAILED",
    "REPLAN_STARTED",
    "REPLAN_FINISHED",
    "FINAL_REVIEW_STARTED",
]

PROGRESS_STAGES = frozenset(
    {
        "PLAN_CREATED",
        "STEP_STARTED",
        "CAPABILITY_SELECTED",
        "STEP_COMPLETED",
        "STEP_PARTIAL",
        "STEP_BLOCKED",
        "STEP_FAILED",
        "REPLAN_STARTED",
        "REPLAN_FINISHED",
        "FINAL_REVIEW_STARTED",
    }
)

PROGRESS_OBJECTIVE_MAX_CHARS = 80
PROGRESS_MESSAGE_MAX_CHARS = 500

# 一旦出现这些标记，就不尝试局部清理，直接替换为通用进度。
FORBIDDEN_PROGRESS_MARKERS = (
    "system prompt",
    "系统提示词",
    "chain of thought",
    "hidden reasoning",
    "隐藏推理",
    "json schema",
    "supervisordecision",
    "planningstate",
    "planning run id",
    "checkpoint thread",
    "traceback (most recent call last)",
)

URL_PATTERN = re.compile(
    r"https?://[^\s<>\]\)\"']+",
    flags=re.IGNORECASE,
)

SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?:[A-Z0-9]+[_-])*(?:"
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|token|"
    r"app[_-]?id|app[_-]?secret|client[_-]?secret|authorization"
    r")\b\s*[:=]\s*(?:bearer\s+)?[\"']?[^,\s;}\]\"']+"
)

SECRET_VALUE_PATTERN = re.compile(
    r"(?i)\b(?:sk|rk|pk)-[A-Za-z0-9_-]{12,}\b"
)

INTERNAL_IDENTIFIER_PATTERN = re.compile(
    r"(?i)\b(?:thread[_ -]?id|planning[_ -]?run[_ -]?id|"
    r"checkpoint[_ -]?id)\b\s*[:=]\s*[^,\s;}\]]+"
)

INTERNAL_AGENT_PATH_PATTERN = re.compile(
    r"(?i)(?:[A-Z]:[\\/][^\s]*[\\/])?"
    r"\.agent(?:[\\/][^\s]*)?"
)

WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)\b[A-Z]:[\\/]"
    r"(?:[^\\/\s]+[\\/])*"
    r"[^\\/\s]*"
)

JSON_LIKE_BLOCK_PATTERN = re.compile(
    r"(?:\{.{80,}\}|\[.{80,}\])",
    flags=re.DOTALL,
)


ProgressCallback = Callable[
    ["ProgressEvent"],
    Awaitable[None],
]


@dataclass(
    frozen=True,
    slots=True,
)
class ProgressEvent:
    """当前用户请求中的高层进度事件。

    这个对象只用于运行时通知：

    - 不进入 PlanningState；
    - 不进入 Conversation messages；
    - 不进入 StepReport；
    - 不计入模型或工具预算。
    """

    stage: ProgressStage
    message: str

    step_id: int | None = None
    total_steps: int | None = None

    capability: str | None = None
    status: str | None = None

    def __post_init__(
        self,
    ) -> None:
        normalized_stage = (
            str(
                self.stage
            )
            .strip()
            .upper()
        )

        normalized_message = (
            str(
                self.message
            )
            .strip()
        )

        if normalized_stage not in PROGRESS_STAGES:
            raise ValueError(
                "未知 ProgressEvent stage："
                f"{self.stage!r}"
            )

        if not normalized_message:
            raise ValueError(
                "ProgressEvent.message 不能为空。"
            )

        if (
            self.step_id is not None
            and self.step_id < 1
        ):
            raise ValueError(
                "ProgressEvent.step_id 不能小于 1。"
            )

        if (
            self.total_steps is not None
            and self.total_steps < 1
        ):
            raise ValueError(
                "ProgressEvent.total_steps 不能小于 1。"
            )

        object.__setattr__(
            self,
            "stage",
            normalized_stage,
        )

        object.__setattr__(
            self,
            "message",
            normalized_message,
        )

        if self.capability is not None:
            object.__setattr__(
                self,
                "capability",
                (
                    str(
                        self.capability
                    )
                    .strip()
                    .upper()
                ),
            )

        if self.status is not None:
            object.__setattr__(
                self,
                "status",
                (
                    str(
                        self.status
                    )
                    .strip()
                    .upper()
                ),
            )

    @property
    def dedup_key(
        self,
    ) -> tuple[
        str,
        int | None,
        str | None,
    ]:
        """返回当前请求内用于防刷屏的稳定Key。

        CAPABILITY_SELECTED使用capability区分能力类别。

        Step开始事件使用status区分不同Attempt，
        防止同一个Step的重试消息被误判为重复事件。
        """

        dedup_variant = (
            self.capability
            or self.status
        )

        return (
            self.stage,
            self.step_id,
            dedup_variant,
        )

def _url_to_domain(
    match: re.Match[str],
) -> str:
    """把完整 URL 转换成域名。"""

    try:
        hostname = (
            urlsplit(
                match.group(0)
            ).hostname
            or ""
        ).strip()

    except ValueError:
        hostname = ""

    return (
        hostname
        or "网页来源"
    )


def _absolute_path_to_name(
    match: re.Match[str],
) -> str:
    """把非 workspace 绝对路径压缩成文件名。"""

    raw_path = (
        match.group(0)
        .rstrip(
            ".,;:，。；："
        )
    )

    try:
        path_name = (
            PureWindowsPath(
                raw_path
            ).name
        )

    except ValueError:
        path_name = ""

    return (
        path_name
        or "本地文件"
    )


def _replace_workspace_root(
    text: str,
) -> str:
    """把 workspace 绝对路径改成相对显示。"""

    workspace_root = (
        WORKSPACE_ROOT
        .resolve()
    )

    result = text

    for root_text in {
        str(
            workspace_root
        ),
        workspace_root.as_posix(),
    }:
        result = re.sub(
            re.escape(
                root_text
            ),
            "workspace",
            result,
            flags=re.IGNORECASE,
        )

    return result


def sanitize_progress_text(
    text: str,
    max_chars: int = (
        PROGRESS_OBJECTIVE_MAX_CHARS
    ),
) -> str:
    """清理准备进入飞书进度消息的文字，并保留有意义的换行。"""

    if (
        isinstance(
            max_chars,
            bool,
        )
        or not isinstance(
            max_chars,
            int,
        )
        or max_chars < 12
    ):
        raise ValueError(
            "max_chars 必须是"
            "大于等于 12 的整数。"
        )

    normalized_text = (
        str(
            text
            or ""
        )
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .strip()
    )

    if not normalized_text:
        return ""

    folded_text = (
        normalized_text
        .casefold()
    )

    if any(
        marker in folded_text
        for marker in (
            FORBIDDEN_PROGRESS_MARKERS
        )
    ):
        return (
            "正在处理内部执行状态。"
        )

    sanitized_text = (
        normalized_text
        .replace(
            "```",
            " ",
        )
    )

    sanitized_text = (
        JSON_LIKE_BLOCK_PATTERN
        .sub(
            " [内部执行详情已隐藏] ",
            sanitized_text,
        )
    )

    sanitized_text = (
        SECRET_ASSIGNMENT_PATTERN
        .sub(
            "[敏感信息已隐藏]",
            sanitized_text,
        )
    )

    sanitized_text = (
        SECRET_VALUE_PATTERN
        .sub(
            "[敏感信息已隐藏]",
            sanitized_text,
        )
    )

    sanitized_text = (
        INTERNAL_IDENTIFIER_PATTERN
        .sub(
            "[内部标识已隐藏]",
            sanitized_text,
        )
    )

    sanitized_text = (
        _replace_workspace_root(
            sanitized_text
        )
    )

    sanitized_text = (
        INTERNAL_AGENT_PATH_PATTERN
        .sub(
            "[内部路径已隐藏]",
            sanitized_text,
        )
    )

    sanitized_text = (
        URL_PATTERN
        .sub(
            _url_to_domain,
            sanitized_text,
        )
    )

    sanitized_text = (
        WINDOWS_ABSOLUTE_PATH_PATTERN
        .sub(
            _absolute_path_to_name,
            sanitized_text,
        )
    )

    normalized_lines: list[str] = []

    previous_line_was_blank = False

    for raw_line in (
        sanitized_text
        .split("\n")
    ):
        # 只压缩当前行内部的空白，
        # 不再把所有换行一起消灭。
        normalized_line = " ".join(
            raw_line.split()
        )

        if normalized_line:
            normalized_lines.append(
                normalized_line
            )

            previous_line_was_blank = False

            continue

        # 多个连续空行最多保留一个，
        # 避免进度消息占用太多屏幕空间。
        if (
            normalized_lines
            and not previous_line_was_blank
        ):
            normalized_lines.append(
                ""
            )

            previous_line_was_blank = True

    sanitized_text = (
        "\n".join(
            normalized_lines
        )
        .strip()
    )

    if not sanitized_text:
        return ""

    if len(
        sanitized_text
    ) <= max_chars:
        return sanitized_text

    return (
        sanitized_text[
            :max_chars - 1
        ].rstrip()
        + "…"
    )

def render_progress_event(
    event: ProgressEvent,
) -> str:
    """生成最终允许发送到飞书的进度文字。"""

    return sanitize_progress_text(
        event.message,
        max_chars=(
            PROGRESS_MESSAGE_MAX_CHARS
        ),
    )


async def safe_emit_progress(
    progress_callback: (
        ProgressCallback
        | None
    ),
    event: ProgressEvent,
) -> bool:
    """发送进度事件，并把发送失败隔离在主任务之外。"""

    if progress_callback is None:
        return False

    try:
        sanitized_event = replace(
            event,
            message=(
                render_progress_event(
                    event
                )
            ),
        )

        await progress_callback(
            sanitized_event
        )

    except Exception:
        # 不记录 event.message，避免敏感内容
        # 在 Warning 日志中再次暴露。
        logger.warning(
            "飞书进度消息发送失败，"
            "Agent 主任务将继续执行。",
            exc_info=False,
        )

        return False

    return True