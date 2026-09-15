"""Single-consumer execution pump for persisted Agent input events."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable

from eventing.models import AgentEvent, EventAction, EventStatus, RunStatus
from eventing.store import AsyncEventStore


logger = logging.getLogger("agent")


class EventRunCancelled(BaseException):
    """Internal control-flow signal raised only at a committed safe point."""

    def __init__(self, cancel_event_id: str) -> None:
        super().__init__(f"Run cancelled by {cancel_event_id}")
        self.cancel_event_id = cancel_event_id


class EventRunReplaced(BaseException):
    """Internal signal that the current execution generation was superseded."""

    def __init__(self, replacement_event_id: str) -> None:
        super().__init__(f"Run replaced by {replacement_event_id}")
        self.replacement_event_id = replacement_event_id

class EventPauseControl:
    """Cooperative rendezvous between one running graph and the event pump."""

    def __init__(self) -> None:
        self._requested = asyncio.Event()
        self._paused = asyncio.Event()
        self._resume = asyncio.Event()
        self._cancel_event_id: str | None = None
        self._replacement_event_id: str | None = None
        self._terminate_requested = False

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    @property
    def cancel_event_id(self) -> str | None:
        return self._cancel_event_id

    @property
    def replacement_event_id(self) -> str | None:
        return self._replacement_event_id

    def request(self) -> None:
        """Request the existing INSERT pause/resume rendezvous."""

        if self._cancel_event_id is not None or self._replacement_event_id is not None:
            return
        self._requested.set()

    def request_cancel(self, cancel_event_id: str) -> None:
        """Request terminal cancellation at the next safe point."""

        normalized = str(cancel_event_id).strip()
        if not normalized:
            raise ValueError("cancel_event_id cannot be empty")
        self._cancel_event_id = normalized
        self._requested.set()

    def request_replace(self, replacement_event_id: str) -> None:
        """Request generation replacement at the next committed safe point."""

        normalized = str(replacement_event_id).strip()
        if not normalized:
            raise ValueError("replacement_event_id cannot be empty")
        if self._cancel_event_id is not None:
            return
        self._replacement_event_id = normalized
        self._requested.set()

    async def wait_until_paused(self) -> None:
        await self._paused.wait()

    async def pause_point(self, *, on_pause=None, on_resume=None) -> bool:
        """Wait only when INSERT has requested the next safe boundary."""

        if not self._requested.is_set():
            return False
        if on_pause is not None:
            paused_result = on_pause()
            if inspect.isawaitable(paused_result):
                await paused_result
        self._paused.set()
        await self._resume.wait()
        terminate_requested = self._terminate_requested
        if not terminate_requested and on_resume is not None:
            resumed_result = on_resume()
            if inspect.isawaitable(resumed_result):
                await resumed_result
        cancel_event_id = self._cancel_event_id
        replacement_event_id = self._replacement_event_id
        self._requested.clear()
        self._paused.clear()
        self._resume.clear()
        self._cancel_event_id = None
        self._replacement_event_id = None
        self._terminate_requested = False
        if terminate_requested:
            if replacement_event_id is not None:
                raise EventRunReplaced(replacement_event_id)
            raise EventRunCancelled(cancel_event_id or "unknown-cancel-event")
        return True

    def resume(self) -> None:
        self._resume.set()

    def terminate(self) -> None:
        """Release a paused handler and make it unwind instead of resuming."""

        self._terminate_requested = True
        self._resume.set()


EventHandler = Callable[[AgentEvent, EventPauseControl, bool], Awaitable[str]]
EventResultHandler = Callable[[AgentEvent, str], Awaitable[None]]
EventFailureHandler = Callable[[AgentEvent, BaseException], Awaitable[None]]
EventCancelHandler = Callable[[AgentEvent, AgentEvent], Awaitable[None]]
EventReplaceHandler = Callable[[AgentEvent, AgentEvent], Awaitable[None]]


class EventRunPump:
    """Claim persisted events one at a time and drive their run lifecycle.

    Channel adapters enqueue quickly and return.  This pump is the only
    consumer, so one personal Agent still executes one top-level user request
    at a time while later messages can be durably accepted.
    """

    def __init__(
        self,
        store: AsyncEventStore,
        *,
        handler: EventHandler,
        on_result: EventResultHandler,
        on_failure: EventFailureHandler,
        on_cancel: EventCancelHandler | None = None,
        on_replace: EventReplaceHandler | None = None,
    ) -> None:
        self.store = store
        self.handler = handler
        self.on_result = on_result
        self.on_failure = on_failure
        self.on_cancel = on_cancel
        self.on_replace = on_replace
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._active_event_id: str | None = None
        self._active_control: EventPauseControl | None = None
        self._active_can_pause = False
        self._pause_targets: set[str] = set()
        self._cancel_targets: dict[str, str] = {}
        self._replace_targets: dict[str, str] = {}

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def active_event_id(self) -> str | None:
        return self._active_event_id

    def start(self) -> None:
        if self.running:
            return
        self._stopping = False
        self._task = asyncio.create_task(
            self._run(),
            name="agent-event-run-pump",
        )
        # Also inspect events persisted before this process started.
        self._wake.set()

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        self._stopping = True
        self._wake.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def notify(self, event: AgentEvent | None = None) -> None:
        """Wake the consumer after the caller durably inserts an event."""

        if not self.running:
            raise RuntimeError("Event Run Pump is not running.")
        if (
            event is not None
            and event.action is EventAction.INSERT
            and event.target_event_id
        ):
            self._pause_targets.add(event.target_event_id)
            if (
                self._active_can_pause
                and self._active_event_id == event.target_event_id
                and self._active_control is not None
            ):
                self._active_control.request()
        elif (
            event is not None
            and event.action is EventAction.CANCEL
            and event.target_event_id
        ):
            self._cancel_targets[event.target_event_id] = event.event_id
            if (
                self._active_event_id == event.target_event_id
                and self._active_control is not None
            ):
                self._active_control.request_cancel(event.event_id)
        elif (
            event is not None
            and event.action is EventAction.REPLACE
            and event.target_event_id
        ):
            self._replace_targets[event.target_event_id] = event.event_id
            if (
                self._active_event_id == event.target_event_id
                and self._active_control is not None
            ):
                self._active_control.request_replace(event.event_id)
        self._wake.set()

    async def _run(self) -> None:
        recovered = await self.store.recover_interrupted_runs()
        if recovered:
            logger.warning(
                "恢复上次进程中断的Agent事件 | event_ids=%s",
                ",".join(recovered),
            )
        while not self._stopping:
            event = await self.store.claim_next_pending()
            if event is None:
                # Clear before checking a second time.  This closes the race in
                # which a producer inserts between the first query and wait().
                self._wake.clear()
                event = await self.store.claim_next_pending()
                if event is None:
                    await self._wake.wait()
                    continue
            if event.action is EventAction.CANCEL:
                await self._apply_pending_cancel(event)
                continue
            if event.action is EventAction.REPLACE:
                await self._apply_pending_replace(event)
                continue
            self._active_event_id = event.event_id
            try:
                await self._execute(event, allow_insert_pause=True)
            finally:
                self._active_event_id = None

    async def _execute(
        self,
        event: AgentEvent,
        *,
        allow_insert_pause: bool,
    ) -> None:
        previous_run = await self.store.require_run(event.event_id)
        resume_from_checkpoint = previous_run.status is RunStatus.PAUSED
        await self.store.update_run_status(event.event_id, RunStatus.RUNNING)
        control = EventPauseControl()
        previous_control = self._active_control
        previous_can_pause = self._active_can_pause
        self._active_control = control
        self._active_can_pause = allow_insert_pause
        pending_cancel_id = self._cancel_targets.get(event.event_id)
        if pending_cancel_id is not None:
            control.request_cancel(pending_cancel_id)
        elif event.event_id in self._replace_targets:
            control.request_replace(self._replace_targets[event.event_id])
        elif allow_insert_pause and event.event_id in self._pause_targets:
            control.request()
        handler_task = asyncio.create_task(
            self.handler(event, control, resume_from_checkpoint),
            name=f"agent-event-handler:{event.event_id}",
        )
        try:
            while not handler_task.done():
                paused_wait = asyncio.create_task(control.wait_until_paused())
                done, _ = await asyncio.wait(
                    {handler_task, paused_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if handler_task in done:
                    paused_wait.cancel()
                    try:
                        await paused_wait
                    except asyncio.CancelledError:
                        pass
                    break

                cancel_event_id = control.cancel_event_id
                if cancel_event_id is not None:
                    application = await self.store.apply_cancel(cancel_event_id)
                    self._cancel_targets.pop(event.event_id, None)
                    self._pause_targets.discard(event.event_id)
                    control.terminate()
                    try:
                        await handler_task
                    except EventRunCancelled:
                        pass
                    except BaseException:
                        logger.exception(
                            "Agent任务在CANCEL收尾时异常退出 | event_id=%s",
                            event.event_id,
                        )
                    cleanup_error = await self._finalize_cancelled_target(
                        application.target_event,
                        application.command_event,
                    )
                    await self._deliver_cancel_result(
                        application.command_event,
                        cleanup_error=cleanup_error,
                    )
                    return

                replacement_event_id = control.replacement_event_id
                if replacement_event_id is not None:
                    application = await self.store.apply_replace(
                        replacement_event_id
                    )
                    self._replace_targets.pop(event.event_id, None)
                    self._pause_targets.discard(event.event_id)
                    control.terminate()
                    try:
                        await handler_task
                    except EventRunReplaced:
                        pass
                    except BaseException:
                        logger.exception(
                            "Agent任务在REPLACE收尾时异常退出 | event_id=%s",
                            event.event_id,
                        )
                    await self._finalize_superseded_target(
                        application.target_event,
                        application.replacement_event,
                    )
                    self._active_event_id = application.replacement_event.event_id
                    await self._execute(
                        application.replacement_event,
                        allow_insert_pause=True,
                    )
                    return

                await self.store.update_run_status(
                    event.event_id,
                    RunStatus.PAUSED,
                )
                inserted = await self.store.claim_pending_insert(
                    target_event_id=event.event_id,
                )
                if inserted is not None:
                    self._active_event_id = inserted.event_id
                    await self._execute(
                        inserted,
                        # The first version intentionally avoids an INSERT
                        # stack. A later INSERT remains durable and is handled
                        # after the suspended parent resumes.
                        allow_insert_pause=False,
                    )
                    self._active_event_id = event.event_id
                self._pause_targets.discard(event.event_id)
                await self.store.update_run_status(
                    event.event_id,
                    RunStatus.RUNNING,
                )
                self._active_control = control
                self._active_can_pause = allow_insert_pause
                control.resume()

            result = await handler_task
        except asyncio.CancelledError as error:
            if not handler_task.done():
                handler_task.cancel()
                try:
                    await handler_task
                except asyncio.CancelledError:
                    pass
            await self._mark_interrupted(event)
            raise
        except BaseException as error:
            await self._mark_failed(event, error, result_code="RUN_FAILED")
            return
        finally:
            self._active_control = previous_control
            self._active_can_pause = previous_can_pause

        await self.store.update_run_status(event.event_id, RunStatus.COMPLETED)
        await self.store.update_event_status(
            event.event_id,
            EventStatus.APPLIED,
            result_code="RUN_COMPLETED",
        )
        try:
            await self.on_result(event, result)
        except Exception:
            # Delivery failure must not rewrite a successfully completed Agent
            # run.  The channel layer may add durable outbox retries later.
            logger.exception(
                "Agent事件已经完成，但结果发送失败 | event_id=%s",
                event.event_id,
            )

    async def _apply_pending_cancel(self, event: AgentEvent) -> None:
        """Consume a CANCEL whose target is no longer the active handler."""

        try:
            application = await self.store.apply_cancel(event.event_id)
        except BaseException as error:
            await self.store.update_event_status(
                event.event_id,
                EventStatus.FAILED,
                result_code="CANCEL_APPLY_FAILED",
            )
            try:
                await self.on_failure(event, error)
            except Exception:
                logger.exception(
                    "CANCEL失败，而且失败通知发送失败 | event_id=%s",
                    event.event_id,
                )
            return
        self._cancel_targets.pop(application.target_event.event_id, None)
        cleanup_error = None
        if application.target_was_cancelled:
            cleanup_error = await self._finalize_cancelled_target(
                application.target_event,
                application.command_event,
            )
        await self._deliver_cancel_result(
            application.command_event,
            cleanup_error=cleanup_error,
        )

    async def _apply_pending_replace(self, event: AgentEvent) -> None:
        """Apply a recovered/queued REPLACE before an orphaned target resumes."""

        try:
            application = await self.store.apply_replace(event.event_id)
        except BaseException as error:
            await self._mark_failed(
                event,
                error,
                result_code="REPLACE_APPLY_FAILED",
            )
            return
        self._replace_targets.pop(application.target_event.event_id, None)
        if application.target_was_superseded:
            await self._finalize_superseded_target(
                application.target_event,
                application.replacement_event,
            )
        self._active_event_id = application.replacement_event.event_id
        await self._execute(
            application.replacement_event,
            allow_insert_pause=True,
        )
        self._active_event_id = None

    async def _finalize_superseded_target(
        self,
        target_event: AgentEvent,
        replacement_event: AgentEvent,
    ) -> None:
        if self.on_replace is None:
            return
        try:
            await self.on_replace(target_event, replacement_event)
        except Exception:
            # The durable generation switch has already committed. Preserve
            # the old resources for recovery, but do not resurrect stale work
            # or prevent the user's replacement task from starting.
            logger.exception(
                "REPLACE已生效，但旧资源归档/清理未完成 | target_event_id=%s",
                target_event.event_id,
            )

    async def _finalize_cancelled_target(
        self,
        target_event: AgentEvent,
        command_event: AgentEvent,
    ) -> BaseException | None:
        if self.on_cancel is None:
            return None
        try:
            await self.on_cancel(target_event, command_event)
        except Exception as error:
            # The durable run is already CANCELLED.  Cleanup is deliberately
            # fail-closed: frozen resources and host archives remain available
            # for startup recovery instead of being deleted speculatively.
            logger.exception(
                "CANCEL已生效，但资源归档/清理未完成 | target_event_id=%s",
                target_event.event_id,
            )
            return error
        return None

    async def _deliver_cancel_result(
        self,
        event: AgentEvent,
        *,
        cleanup_error: BaseException | None = None,
    ) -> None:
        if cleanup_error is not None:
            message = (
                "当前任务已停止；部分资源没有完成自动清理，"
                "系统已保留现场，启动恢复时会继续核对。"
            )
        elif event.result_code == "TARGET_CANCELLED":
            message = "当前任务已在安全节点取消，执行记录和归档已保留。"
        else:
            message = "目标任务在取消生效前已经结束，没有重复修改它的状态。"
        try:
            await self.on_result(event, message)
        except Exception:
            logger.exception(
                "CANCEL已经生效，但结果发送失败 | event_id=%s",
                event.event_id,
            )

    async def _mark_failed(
        self,
        event: AgentEvent,
        error: BaseException,
        *,
        result_code: str,
        notify: bool = True,
    ) -> None:
        await self.store.update_run_status(event.event_id, RunStatus.FAILED)
        await self.store.update_event_status(
            event.event_id,
            EventStatus.FAILED,
            result_code=result_code,
        )
        if notify:
            try:
                await self.on_failure(event, error)
            except Exception:
                logger.exception(
                    "Agent事件失败，而且失败通知发送失败 | event_id=%s",
                    event.event_id,
                )

    async def _mark_interrupted(self, event: AgentEvent) -> None:
        """Keep the last committed graph checkpoint resumable on shutdown."""

        run = await self.store.require_run(event.event_id)
        if run.status is RunStatus.RUNNING:
            await self.store.update_run_status(event.event_id, RunStatus.PAUSED)
        stored_event = await self.store.require_event(event.event_id)
        if stored_event.status is EventStatus.HANDLING:
            await self.store.update_event_status(
                event.event_id,
                EventStatus.PENDING,
                result_code="PUMP_STOPPED_RESUME_REQUIRED",
            )


__all__ = ["EventPauseControl", "EventRunCancelled", "EventRunPump"]
