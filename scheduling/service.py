"""Background runner joining the schedule store to Windows notifications."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from scheduling.models import DeliveryTarget, ReminderRun, ReminderRunStatus, ReminderSchedule
from scheduling.store import AsyncScheduleStore
from scheduling.windows_notifications import WindowsToastNotifier


logger = logging.getLogger("agent")

DeliveryHandler = Callable[[ReminderSchedule, ReminderRun], Awaitable[dict[str, Any]]]
ScheduleContextResolver = Callable[[str], Awaitable[tuple[str, str]]]


class ScheduleService:
    def __init__(
        self,
        store: AsyncScheduleStore | None = None,
        notifier: WindowsToastNotifier | None = None,
        *,
        poll_seconds: float = 15.0,
        dispatch_enabled: bool = True,
        context_resolver: ScheduleContextResolver | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds必须大于0。")
        self.store = store or AsyncScheduleStore()
        self.notifier = notifier or WindowsToastNotifier()
        self.poll_seconds = poll_seconds
        self._wake = asyncio.Event()
        self._dispatch_ready = asyncio.Event()
        if dispatch_enabled:
            self._dispatch_ready.set()
        self._context_resolver = context_resolver
        self._delivery_handlers: dict[DeliveryTarget, DeliveryHandler] = {
            DeliveryTarget.WINDOWS: self._deliver_windows,
        }
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        await self.store.start()
        recovered = await self.store.recover_claimed()
        if recovered:
            logger.warning("恢复进程中断前尚未确认提交的本地提醒 | count=%s", recovered)
        self._task = asyncio.create_task(self._run(), name="local-schedule-runner")
        self._wake.set()

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.store.close()

    def notify_changed(self) -> None:
        if self.running:
            self._wake.set()

    def configure_delivery_handler(
        self,
        target: DeliveryTarget,
        handler: DeliveryHandler | None,
    ) -> None:
        if target is DeliveryTarget.WINDOWS:
            raise ValueError("Windows投递器由ScheduleService内部管理。")
        if handler is None:
            self._delivery_handlers.pop(target, None)
        else:
            self._delivery_handlers[target] = handler

    def enable_dispatch(self) -> None:
        self._dispatch_ready.set()
        self.notify_changed()

    def disable_dispatch(self) -> None:
        self._dispatch_ready.clear()

    async def resolve_context(self, event_id: str) -> tuple[str, str]:
        normalized_event_id = str(event_id).strip()
        if not normalized_event_id or self._context_resolver is None:
            raise RuntimeError("当前调用缺少可验证的飞书会话上下文。")
        return await self._context_resolver(normalized_event_id)

    async def _deliver_windows(
        self,
        schedule: ReminderSchedule,
        run: ReminderRun,
    ) -> dict[str, Any]:
        receipt = await self.notifier.send(
            title=schedule.title,
            message=schedule.message,
            tag=schedule.schedule_id,
        )
        return {
            "accepted_by_windows": receipt.accepted_by_windows,
            "provider": receipt.provider,
            "detail": receipt.detail,
        }

    async def submit_notification(self, *, title: str, message: str, tag: str = "") -> dict:
        receipt = await self.notifier.send(title=title, message=message, tag=tag)
        return {
            "accepted_by_windows": receipt.accepted_by_windows,
            "provider": receipt.provider,
            "detail": receipt.detail,
        }

    async def run_once(self) -> int:
        await self.store.materialize_due()
        submitted = 0
        while True:
            if not self._dispatch_ready.is_set():
                return submitted
            claimed = await self.store.claim_next()
            if claimed is None:
                return submitted
            schedule, run = claimed
            try:
                handler = self._delivery_handlers.get(schedule.delivery_target)
                if handler is None:
                    raise RuntimeError(
                        f"未配置{schedule.delivery_target.value}日程投递器。"
                    )
                await handler(schedule, run)
            except Exception as error:
                await self.store.finish_run(
                    run.run_id,
                    ReminderRunStatus.FAILED,
                    error=str(error)[:2000],
                )
                logger.exception(
                    "日程投递失败 | target=%s schedule_id=%s run_id=%s",
                    schedule.delivery_target.value,
                    schedule.schedule_id,
                    run.run_id,
                )
            else:
                await self.store.finish_run(run.run_id, ReminderRunStatus.SUBMITTED)
                submitted += 1

    async def _run(self) -> None:
        while True:
            try:
                await self._dispatch_ready.wait()
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("本地提醒扫描失败；将在下一轮继续")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass
