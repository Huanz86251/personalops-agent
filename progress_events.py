from __future__ import annotations

import logging
import re
import asyncio
from time import monotonic
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import PureWindowsPath
from typing import Literal, TypeVar
from urllib.parse import urlsplit

from langchain_core.callbacks import AsyncCallbackHandler

from path import WORKSPACE_ROOT


logger = logging.getLogger("agent")


ProgressStage = Literal[
    "PLAN_CREATED",
    "STEP_STARTED",
    "CAPABILITY_SELECTED",
    "MODEL_STARTED",
    "TOOL_STARTED",
    "ACTIVITY_WAITING",
    "STEP_REVIEW_STARTED",
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
        "MODEL_STARTED",
        "TOOL_STARTED",
        "ACTIVITY_WAITING",
        "STEP_REVIEW_STARTED",
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

_ResultT = TypeVar("_ResultT")


class OperationInactivityTimeoutError(TimeoutError):
    """The top-level run stopped producing real runtime activity."""

    def __init__(self, *, silent_seconds: float, timeout_seconds: float) -> None:
        self.silent_seconds = float(silent_seconds)
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(
            "Agent连续"
            f"{int(self.silent_seconds)}秒没有模型、工具、规划或审核活动，"
            "已按静默卡死处理。"
        )


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

    except Exception as error:
        # 不记录 event.message，避免敏感内容
        # 在 Warning 日志中再次暴露。
        logger.warning(
            "飞书进度消息发送失败，"
            "Agent 主任务将继续执行 | error_type=%s",
            type(error).__name__,
            exc_info=False,
        )

        return False

    return True


async def run_with_progress_keepalive(
    operation: Callable[[ProgressCallback], Awaitable[_ResultT]],
    progress_callback: ProgressCallback,
    *,
    interval_seconds: float = 30.0,
    waiting_for_user: Callable[[], bool] | None = None,
    inactivity_timeout_seconds: float | None = None,
    code_inactivity_timeout_seconds: float | None = None,
) -> _ResultT:
    """Keep Feishu alive and stop only a genuinely silent top-level run.

    The timer only reports a gap since the last real progress event. It never
    claims that a particular tool is still running, and is stopped before the
    final answer can be delivered. The optional watchdog is inactivity based,
    not a total runtime limit: every real model/tool/planning/review event
    resets it, and waiting for a user answer suspends it. Once Code activity is
    observed, the longer Code silence allowance remains active for the run.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds 必须大于零。")
    if inactivity_timeout_seconds is not None and inactivity_timeout_seconds <= 0:
        raise ValueError("inactivity_timeout_seconds 必须大于零。")
    if (
        code_inactivity_timeout_seconds is not None
        and code_inactivity_timeout_seconds <= 0
    ):
        raise ValueError("code_inactivity_timeout_seconds 必须大于零。")

    last_real_event_at = monotonic()
    pulse_count = 0
    code_activity_seen = False

    def effective_inactivity_timeout() -> float | None:
        if code_activity_seen and code_inactivity_timeout_seconds is not None:
            return code_inactivity_timeout_seconds
        return inactivity_timeout_seconds

    async def observed_progress(event: ProgressEvent) -> None:
        nonlocal last_real_event_at, code_activity_seen
        # Keepalive pulses are display-only and must never keep a stuck run
        # alive. All other events are emitted by real Harness/model/tool state.
        if event.stage != "ACTIVITY_WAITING":
            last_real_event_at = monotonic()
            if str(event.capability or "").upper() in {
                "CODE",
                "CODE_AGENT",
                "CODE_REVIEWER",
                "REVIEWER",
            }:
                code_activity_seen = True
        await progress_callback(event)

    async def keepalive() -> None:
        nonlocal last_real_event_at, pulse_count
        while True:
            await asyncio.sleep(interval_seconds)
            # A Step-level question is a deliberate pause, not silent work.
            # The question is sent separately; do not claim the Agent is still
            # processing while it is actually waiting for the user's answer.
            if waiting_for_user is not None and waiting_for_user():
                last_real_event_at = monotonic()
                continue
            silent_seconds = monotonic() - last_real_event_at
            if silent_seconds < interval_seconds:
                continue
            pulse_count += 1
            await safe_emit_progress(
                progress_callback,
                ProgressEvent(
                    stage="ACTIVITY_WAITING",
                    message=(
                        "仍在处理当前任务，"
                        f"约 {int(silent_seconds)} 秒没有新的执行进度。"
                    ),
                    status=f"pulse-{pulse_count}",
                ),
            )

    async def inactivity_watchdog(operation_task: asyncio.Task[_ResultT]) -> None:
        nonlocal last_real_event_at
        while not operation_task.done():
            timeout = effective_inactivity_timeout()
            if timeout is None:
                # Keep the helper alive until the operation finishes. Returning
                # here would incorrectly win FIRST_COMPLETED when the watchdog
                # is intentionally disabled (for example in focused tests).
                await asyncio.shield(operation_task)
                return
            await asyncio.sleep(min(30.0, max(0.01, timeout / 4.0)))
            if operation_task.done():
                return
            if waiting_for_user is not None and waiting_for_user():
                # Human response latency is not Agent inactivity.
                last_real_event_at = monotonic()
                continue
            silent_seconds = monotonic() - last_real_event_at
            timeout = effective_inactivity_timeout()
            if timeout is None or silent_seconds < timeout:
                continue
            logger.error(
                "Agent连续无真实活动，触发静默卡死保护 | "
                "silent_seconds=%s | timeout_seconds=%s | code_activity=%s",
                int(silent_seconds),
                int(timeout),
                code_activity_seen,
            )
            raise OperationInactivityTimeoutError(
                silent_seconds=silent_seconds,
                timeout_seconds=timeout,
            )

    # This task is created before entering LangGraph; creating it from a
    # Worker callback would make graph execution wait on the timer itself.
    keepalive_task = asyncio.create_task(keepalive())
    operation_task = asyncio.create_task(operation(observed_progress))
    watchdog_task = asyncio.create_task(inactivity_watchdog(operation_task))
    try:
        done, _ = await asyncio.wait(
            {operation_task, watchdog_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return await operation_task

        # The watchdog ended first. Its exception is the terminal diagnosis;
        # cancellation only stops the abandoned business coroutine.
        watchdog_error = watchdog_task.exception()
        operation_task.cancel()
        await asyncio.wait({operation_task}, timeout=5.0)
        if watchdog_error is not None:
            raise watchdog_error
        raise RuntimeError("Agent静默监控意外结束。")
    finally:
        keepalive_task.cancel()
        watchdog_task.cancel()
        if not operation_task.done():
            operation_task.cancel()
        finished_tasks = [keepalive_task, watchdog_task]
        if operation_task.done():
            finished_tasks.append(operation_task)
        else:
            logger.error(
                "静默超时后业务协程仍未响应取消；退出流程将依靠进程级保险。"
            )
        await asyncio.gather(*finished_tasks, return_exceptions=True)


_HIDDEN_PROGRESS_TOOLS = frozenset({"publish_worker_progress"})
_TOOL_PROGRESS_LABELS = {
    "execute": "正在运行代码或测试",
    "read_file": "正在读取工作文件",
    "write_file": "正在写入工作文件",
    "fetch_webpage": "正在读取网页",
    "list_city_movies": "正在查询当地电影",
    "get_movie_showtimes": "正在查询电影场次",
    "get_cinema_showtimes": "正在查询影院场次",
    "list_cinemas": "正在查询当地影院",
    "send_local_file_to_feishu": "正在发送文件",
    "appworld_discover": "正在查看 AppWorld 接口文档",
    "appworld_execute": "正在调用 AppWorld 接口",
    "appworld_verify": "正在核验 AppWorld 结果",
    "submit_code_for_review": "正在提交代码供审核",
    "submit_code_review": "正在提交代码审核结论",
    "report_general_result": "正在提交执行结果",
}


def tool_progress_label(name: str) -> str:
    """Only disclose a tool category/name, never its arguments or response."""
    normalized = str(name or "").strip()
    if normalized in _TOOL_PROGRESS_LABELS:
        return _TOOL_PROGRESS_LABELS[normalized]
    lowered = normalized.lower()
    if "search" in lowered:
        return "正在搜索外部信息"
    if "browser" in lowered or "playwright" in lowered:
        return "正在浏览网页"
    if "file" in lowered:
        return "正在处理文件"
    # Unknown names might originate from an external tool registry. Do not
    # publish their raw names before they have an explicit display mapping.
    return "正在调用工作工具"


class LiveToolProgressHandler(AsyncCallbackHandler):
    """Emit bounded, runtime-observed tool starts for one Worker invocation."""

    run_inline = True

    def __init__(
        self,
        progress_callback: ProgressCallback,
        *,
        step_id: int | None,
        worker_id: str,
        worker_role: str = "",
    ) -> None:
        self.progress_callback = progress_callback
        self.step_id = step_id
        self.worker_id = worker_id
        self.worker_role = str(worker_role or "").strip().upper() or None
        self.count = 0
        self._last_by_name: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def on_chat_model_start(
        self, serialized, messages, *, run_id, **kwargs
    ) -> None:
        async with self._lock:
            self.count += 1
            sequence = self.count
        await safe_emit_progress(
            self.progress_callback,
            ProgressEvent(
                stage="MODEL_STARTED",
                message="Worker 正在分析当前步骤并准备下一步。",
                step_id=self.step_id,
                capability=self.worker_role,
                status=f"{self.worker_id}:{sequence}",
            ),
        )

    async def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id,
        **kwargs,
    ) -> None:
        name = str((serialized or {}).get("name") or "").strip()
        if not name or name in _HIDDEN_PROGRESS_TOOLS:
            return
        async with self._lock:
            self.count += 1
            sequence = self.count
            now = monotonic()
            # Frequent identical calls should not saturate Feishu edits.
            if now - self._last_by_name.get(name, float("-inf")) < 2.0:
                return
            self._last_by_name[name] = now
            event = ProgressEvent(
                stage="TOOL_STARTED",
                message=tool_progress_label(name) + "。",
                step_id=self.step_id,
                capability=self.worker_role,
                status=f"{self.worker_id}:{sequence}",
            )
        await safe_emit_progress(self.progress_callback, event)
