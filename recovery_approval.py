"""Harness-owned human confirmation for report-proposed task downgrades.

A Reviewer, or a Web Worker closing its own Step, may describe one exact weaker
outcome, but it cannot approve the change and the Scheduler cannot infer
approval. This broker pauses the active run until the authenticated channel
returns an explicit allow/refuse answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecoveryApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    chat_id: str = ""
    step_id: int = Field(ge=1)
    worker_kind: Literal["WEB", "GENERAL", "CODE"]
    original_user_request: str = Field(min_length=1, max_length=12000)
    current_step: str = Field(min_length=1, max_length=6000)
    downgrade_reason: str = Field(min_length=1, max_length=4000)
    proposed_degraded_outcome: str = Field(min_length=1, max_length=6000)
    user_confirmation_request: str = Field(min_length=1, max_length=6000)

    @property
    def decision_key(self) -> tuple[str, int, str]:
        payload = json.dumps(
            {
                "reason": self.downgrade_reason,
                "outcome": self.proposed_degraded_outcome,
                "question": self.user_confirmation_request,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            self.event_id,
            self.step_id,
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )


class RecoveryHumanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["APPROVED", "DENIED", "UNAVAILABLE"]
    source: Literal["USER", "HARNESS"]
    message: str = Field(min_length=1)


@dataclass(frozen=True)
class HumanReplyResult:
    handled: bool
    reply: str = ""


HumanNotifier = Callable[[str, RecoveryApprovalRequest], Awaitable[None]]


class RecoveryApprovalService:
    def __init__(self) -> None:
        self.human_notifier: HumanNotifier | None = None
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._pending_requests: dict[str, RecoveryApprovalRequest] = {}
        self._decisions: dict[tuple[str, int, str], bool] = {}

    def configure_human_notifier(self, notifier: HumanNotifier | None) -> None:
        self.human_notifier = notifier

    async def request_decision(
        self,
        request: RecoveryApprovalRequest,
    ) -> RecoveryHumanDecision:
        prior = self._decisions.get(request.decision_key)
        if prior is not None:
            return self._decision(prior)
        if not request.chat_id or self.human_notifier is None:
            return RecoveryHumanDecision(
                status="UNAVAILABLE",
                source="HARNESS",
                message=(
                    "当前渠道无法取得用户对降级方案的明确确认；"
                    "Scheduler不得采用该降级。"
                ),
            )
        if request.chat_id in self._pending:
            return RecoveryHumanDecision(
                status="UNAVAILABLE",
                source="HARNESS",
                message=(
                    "当前会话已有一个降级确认等待处理；"
                    "Scheduler不得把未确认方案视为已批准。"
                ),
            )

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
        self._decisions[request.decision_key] = allowed
        return self._decision(allowed)

    @staticmethod
    def _decision(allowed: bool) -> RecoveryHumanDecision:
        if allowed:
            return RecoveryHumanDecision(
                status="APPROVED",
                source="USER",
                message="用户已明确允许本次具体降级方案。",
            )
        return RecoveryHumanDecision(
            status="DENIED",
            source="USER",
            message=(
                "用户已明确拒绝本次具体降级方案；Scheduler只能继续尝试满足"
                "原要求，或如实结束，不能换一种说法再次采用同一降级。"
            ),
        )

    def consume_user_reply(self, chat_id: str, text: str) -> HumanReplyResult:
        future = self._pending.get(chat_id)
        if future is None:
            return HumanReplyResult(False)
        normalized = "".join(str(text).strip().split()).casefold()
        if normalized == "允许":
            if not future.done():
                future.set_result(True)
            return HumanReplyResult(
                True,
                "已允许本次具体降级方案，任务将交回 Scheduler 重新规划。",
            )
        if normalized in {"拒绝", "不允许"}:
            if not future.done():
                future.set_result(False)
            return HumanReplyResult(
                True,
                "已拒绝本次具体降级方案；任务不会把该方案改写成已批准。",
            )
        return HumanReplyResult(
            True,
            "当前正在等待降级决定。请只回复“允许”或“拒绝”。",
        )

    def has_pending(self, chat_id: str | None = None) -> bool:
        if chat_id is None:
            return bool(self._pending)
        return str(chat_id).strip() in self._pending


RECOVERY_APPROVALS = RecoveryApprovalService()


def configure_recovery_human_notifier(notifier: HumanNotifier | None) -> None:
    RECOVERY_APPROVALS.configure_human_notifier(notifier)


__all__ = [
    "RECOVERY_APPROVALS",
    "RecoveryApprovalRequest",
    "RecoveryApprovalService",
    "RecoveryHumanDecision",
    "configure_recovery_human_notifier",
]
