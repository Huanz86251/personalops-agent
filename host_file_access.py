"""Harness-owned boundary for local reads, external disclosure and mutations.

Local discovery/import is read-only and allowed by policy.  A Worker proposing
external disclosure supplies one exact reference and reason; the Harness
combines that proposal with trusted run context before asking a small
independent Judge.  External mutation always goes to the authenticated user.
A refusal is preserved as USER_DENIED rather than being disguised as a missing
file or tool failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from prompt_loader import render_prompt


logger = logging.getLogger(__name__)

_JUDGE_REASON_MAX_CHARS = 300


def _compact_judge_reason(value: Any) -> str:
    """Keep Judge explanations bounded without making parsing fail closed."""

    normalized = " ".join(str(value or "").strip().split())
    if not normalized:
        normalized = "独立文件审批器没有提供可用理由，需要用户明确确认。"
    if len(normalized) <= _JUDGE_REASON_MAX_CHARS:
        return normalized
    return normalized[: _JUDGE_REASON_MAX_CHARS - 1].rstrip() + "…"


def _local_volume_roots() -> tuple[Path, ...]:
    """Return mounted local volumes without relying on model-guessed paths."""

    if os.name == "nt":
        roots = tuple(
            Path(f"{letter}:\\").resolve()
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if Path(f"{letter}:\\").exists()
        )
        return roots
    return (Path("/").resolve(),)


class HostFileUserDenied(PermissionError):
    code = "USER_DENIED"

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__("用户拒绝不允许使用该文件。")


class HostFileAccessDenied(PermissionError):
    code = "POLICY_DENIED"


class HostFileApprovalUnavailable(RuntimeError):
    code = "APPROVAL_UNAVAILABLE"


class HostFileJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["ALLOW", "ASK_USER"] = Field(
        description=(
            "仅当原始请求已清楚授权发送这个精确文件时填ALLOW；任何歧义、范围扩大"
            "或路径对应关系不足时填ASK_USER。"
        ),
    )
    # Do not put the operational display limit into the provider schema. Some
    # providers generate a sound decision with an overlong explanation; the
    # Harness truncates that explanation deterministically after validation.
    reason: str = Field(
        min_length=1,
        description="简要说明授权依据或仍需用户确认的具体歧义；Harness会确定性截断过长理由。",
    )


class HostFileApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(min_length=1)
    chat_id: str = ""
    worker_id: str = Field(min_length=1)
    worker_kind: str = Field(min_length=1)
    operation: Literal[
        "READ_IMPORT",
        "SEND_TO_USER",
        "WRITE_EXTERNAL",
        "MOVE_EXTERNAL",
        "DELETE_EXTERNAL",
    ]
    path: str = Field(min_length=1, max_length=2048)
    worker_reason: str = Field(min_length=1, max_length=1200)
    original_user_request: str = Field(min_length=1, max_length=12000)
    current_step: str = Field(min_length=1, max_length=6000)
    trusted_path_mapping: str = Field(
        default="",
        max_length=6000,
        description=(
            "Harness解析的来源、Worker可见引用与隔离副本之间的可信映射；"
            "不接受Worker自行声明。"
        ),
    )


class HostFilePolicy:
    """Static path limits remain authoritative even when a model says ALLOW."""

    def __init__(self, roots: tuple[Path, ...]) -> None:
        self.roots = tuple(Path(root).resolve() for root in roots)

    @classmethod
    def from_env(cls) -> "HostFilePolicy":
        # Read-only discovery/import is owner-local and does not cross an
        # external boundary.  It therefore defaults to every mounted local
        # volume.  Operators can still opt into a narrower set for deployments
        # that do not share the owner's machine trust model.
        scope = os.environ.get("HOST_FILE_READ_SCOPE", "ALL_LOCAL").strip().upper()
        if scope == "ALL_LOCAL":
            return cls(_local_volume_roots())
        if scope != "CONFIGURED_ROOTS":
            raise ValueError(
                "HOST_FILE_READ_SCOPE 必须是 ALL_LOCAL 或 CONFIGURED_ROOTS。"
            )
        raw = os.environ.get("HOST_FILE_ACCESS_ROOTS", "[]").strip()
        try:
            values = json.loads(raw or "[]")
        except json.JSONDecodeError as error:
            raise ValueError("HOST_FILE_ACCESS_ROOTS 必须是 JSON 数组。") from error
        if not isinstance(values, list) or any(
            not isinstance(item, str) or not item.strip() for item in values
        ):
            raise ValueError("HOST_FILE_ACCESS_ROOTS 必须是非空路径字符串组成的 JSON 数组。")
        roots: list[Path] = []
        for value in values:
            path = Path(value)
            if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
                raise ValueError("主机文件访问根目录必须是本地绝对路径。")
            roots.append(path.resolve())
        return cls(tuple(roots))

    @property
    def enabled(self) -> bool:
        return bool(self.roots)

    def validate(self, raw_path: str, *, require_file: bool = False) -> tuple[Path, Path]:
        path = Path(str(raw_path).strip())
        if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
            raise HostFileAccessDenied("只允许配置根目录内的本地绝对路径。")
        # Alternate data streams remain outside the ordinary file model.  A
        # normal Windows drive prefix is carried by ``path.drive`` rather than
        # a path component, so this does not reject C:/...
        if ":" in path.name:
            raise HostFileAccessDenied("不支持 NTFS 附加数据流。")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise HostFileAccessDenied("请求的主机路径不存在或无法访问。") from error
        root = next((item for item in self.roots if resolved.is_relative_to(item)), None)
        if root is None:
            raise HostFileAccessDenied("该路径不在本机配置的允许访问目录中。")
        # Reject the complete existing chain, not only the final target.
        cursor = resolved
        while cursor.is_relative_to(root):
            info = cursor.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise HostFileAccessDenied("不允许访问符号链接或目录联接。")
            if cursor == root:
                break
            cursor = cursor.parent
        if require_file and not resolved.is_file():
            raise HostFileAccessDenied("读取导入操作只接受普通文件。")
        return resolved, root


@dataclass(frozen=True)
class HumanReplyResult:
    handled: bool
    reply: str = ""


HumanNotifier = Callable[[str, HostFileApprovalRequest], Awaitable[None]]


class HostFileAccessService:
    def __init__(self, policy: HostFilePolicy | None = None) -> None:
        self.policy = policy or HostFilePolicy.from_env()
        self.judge_model: Any | None = None
        self.human_notifier: HumanNotifier | None = None
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._pending_requests: dict[str, HostFileApprovalRequest] = {}
        self._decisions: dict[tuple[str, str, str], bool] = {}

    def configure_judge(self, model: Any | None) -> None:
        self.judge_model = model

    def configure_policy(self, policy: HostFilePolicy) -> None:
        self.policy = policy

    def configure_human_notifier(self, notifier: HumanNotifier | None) -> None:
        self.human_notifier = notifier

    async def _judge(self, request: HostFileApprovalRequest) -> HostFileJudgeOutput:
        if self.judge_model is None:
            return HostFileJudgeOutput(
                decision="ASK_USER",
                reason="独立文件审批模型尚未配置。",
            )
        prompt = render_prompt(
            "security/host_file_judge",
            original_user_request=request.original_user_request,
            current_step=request.current_step,
            worker_kind=request.worker_kind,
            worker_id=request.worker_id,
            operation=request.operation,
            path=request.path,
            worker_reason=request.worker_reason,
            trusted_path_mapping=request.trusted_path_mapping or "未提供可信路径映射。",
        )
        try:
            from structured_output_policy import with_provider_structured_output
            structured = with_provider_structured_output(
                self.judge_model,
                HostFileJudgeOutput,
                include_raw=False,
                fallback_method=None,
            )
            raw = await structured.ainvoke([SystemMessage(content=prompt)])
            judgment = (
                raw
                if isinstance(raw, HostFileJudgeOutput)
                else HostFileJudgeOutput.model_validate(raw)
            )
        except Exception as error:
            # A malformed or unavailable Judge must never turn into an upload
            # tool failure. Escalating to the authenticated user is the safe
            # fallback and preserves the same paused tool call.
            logger.warning(
                "文件外发Judge结果无法解析，安全降级为用户确认 | error=%s: %s",
                type(error).__name__,
                error,
            )
            return HostFileJudgeOutput(
                decision="ASK_USER",
                reason="独立文件审批结果无法可靠解析，需要用户明确确认。",
            )
        return judgment.model_copy(
            update={"reason": _compact_judge_reason(judgment.reason)}
        )

    async def authorize(self, request: HostFileApprovalRequest) -> None:
        raw_path = str(request.path).strip().replace("\\", "/")
        normalized_path = (
            raw_path.rstrip("/") or "/"
            if raw_path.startswith("/")
            else str(Path(request.path).resolve())
        )
        key = (request.event_id, request.operation, normalized_path)
        prior = self._decisions.get(key)
        if prior is True:
            return
        if prior is False:
            raise HostFileUserDenied(request.path)
        # External mutation is never delegated to the small Judge.  A human
        # approval for a directory covers its descendants for this event and
        # operation, avoiding one prompt per child while preserving scope.
        if request.operation in {"WRITE_EXTERNAL", "MOVE_EXTERNAL", "DELETE_EXTERNAL"}:
            for (event_id, operation, approved_path), allowed in self._decisions.items():
                if not allowed or event_id != request.event_id or operation != request.operation:
                    continue
                try:
                    Path(normalized_path).relative_to(Path(approved_path))
                    return
                except ValueError:
                    pass
            judgment = HostFileJudgeOutput(
                decision="ASK_USER",
                reason="工作区之外的修改、覆盖、移动或删除必须由用户明确确认。",
            )
        else:
            judgment = await self._judge(request)
        if judgment.decision == "ALLOW":
            self._decisions[key] = True
            return
        if not request.chat_id or self.human_notifier is None:
            raise HostFileApprovalUnavailable("文件访问需要用户确认，但当前渠道无法发起确认。")
        if request.chat_id in self._pending:
            raise HostFileApprovalUnavailable("当前会话已有一个文件权限请求等待用户处理。")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending[request.chat_id] = future
        self._pending_requests[request.chat_id] = request
        try:
            await self.human_notifier(request.chat_id, request)
            allowed = await future
        finally:
            self._pending.pop(request.chat_id, None)
            self._pending_requests.pop(request.chat_id, None)
        self._decisions[key] = allowed
        if not allowed:
            raise HostFileUserDenied(request.path)

    def consume_user_reply(self, chat_id: str, text: str) -> HumanReplyResult:
        future = self._pending.get(chat_id)
        if future is None:
            return HumanReplyResult(False)
        normalized = "".join(str(text).strip().split()).casefold()
        if normalized == "允许":
            if not future.done():
                future.set_result(True)
            return HumanReplyResult(True, "已允许本次文件操作，任务继续执行。")
        if normalized in {"拒绝", "不允许"}:
            if not future.done():
                future.set_result(False)
            return HumanReplyResult(True, "已拒绝本次文件操作；任务会收到 USER_DENIED，不会把它解释成文件不存在。")
        return HumanReplyResult(True, "当前正在等待文件权限决定。请只回复“允许”或“拒绝”。")

    def has_pending(self, chat_id: str | None = None) -> bool:
        if chat_id is None:
            return bool(self._pending)
        return str(chat_id).strip() in self._pending


HOST_FILE_ACCESS = HostFileAccessService()


def configure_host_file_judge(model: Any | None) -> None:
    HOST_FILE_ACCESS.configure_judge(model)


def configure_host_file_policy(policy: HostFilePolicy) -> None:
    HOST_FILE_ACCESS.configure_policy(policy)
    # Tool schemas are assembled after owner configuration so the Worker sees
    # the exact current roots before its first model round.  Keep the import
    # local to avoid a module cycle while host_file_tools is being defined.
    try:
        from tools.host_file_tools import refresh_host_file_tool_descriptions
    except ImportError:
        return
    refresh_host_file_tool_descriptions()


def configure_host_file_human_notifier(notifier: HumanNotifier | None) -> None:
    HOST_FILE_ACCESS.configure_human_notifier(notifier)


__all__ = [
    "HOST_FILE_ACCESS",
    "HostFileAccessDenied",
    "HostFileApprovalRequest",
    "HostFileApprovalUnavailable",
    "HostFileJudgeOutput",
    "HostFilePolicy",
    "HostFileUserDenied",
    "HumanReplyResult",
    "configure_host_file_human_notifier",
    "configure_host_file_judge",
    "configure_host_file_policy",
]
