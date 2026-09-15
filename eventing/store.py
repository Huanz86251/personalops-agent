"""SQLite persistence for input events and their LangGraph run state."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import aiosqlite

from delivery.models import (
    PromotionStatus,
    WorkspacePromotion,
)
from eventing.models import (
    AgentEvent,
    EventAction,
    EventRun,
    EventStatus,
    RunStatus,
    transition_event_status,
    transition_run_status,
)
from path import AGENT_DATA_ROOT


EVENT_STORE_PATH = AGENT_DATA_ROOT / "events.sqlite3"


class EventStoreNotStartedError(RuntimeError):
    """Raised when a store operation is attempted before start()."""


class EventNotFoundError(LookupError):
    """Raised when an event identifier is not present in the store."""


class EventRunNotFoundError(LookupError):
    """Raised when an event does not own an execution record."""


class EventConflictError(RuntimeError):
    """Raised when persisted state changed before an atomic transition."""


@dataclass(frozen=True, slots=True)
class CancelApplication:
    """Durable result of applying one CANCEL command to its target run."""

    command_event: AgentEvent
    target_event: AgentEvent
    target_run: EventRun
    target_was_cancelled: bool


@dataclass(frozen=True, slots=True)
class ReplaceApplication:
    """Durable handoff from one superseded run to its replacement run."""

    replacement_event: AgentEvent
    replacement_run: EventRun
    target_event: AgentEvent
    target_run: EventRun
    target_was_superseded: bool


class WorkerGroupNotFoundError(LookupError):
    """Raised when a Worker Group identifier is not persisted."""


@dataclass(frozen=True, slots=True)
class StoredWorkerGroup:
    """A durable group snapshot plus store-owned coordination metadata."""

    snapshot: dict[str, Any]
    revision: int
    review_id: str | None
    report: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class WorkerGroupCASResult:
    """Result of an optimistic Worker Group snapshot write."""

    applied: bool
    record: StoredWorkerGroup


@dataclass(frozen=True, slots=True)
class WorkerGroupReviewClaim:
    """Result of trying to become the only Reporter caller for a group."""

    acquired_now: bool
    owned_by_caller: bool
    record: StoredWorkerGroup


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_text(value: datetime | None = None) -> str:
    timestamp = value or _utc_now()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("changed_at must include timezone information.")
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_worker_group_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(snapshot)
    group_id = str(normalized.get("group_id") or "").strip()
    event_id = str(normalized.get("event_id") or "").strip()
    step_id = normalized.get("step_id")
    join_policy = str(normalized.get("join_policy") or "").strip()
    review_status = str(normalized.get("review_status") or "").strip()
    created_at = str(normalized.get("created_at") or "").strip()
    updated_at = str(normalized.get("updated_at") or "").strip()
    slots = normalized.get("slots")

    if not group_id or not event_id:
        raise ValueError("Worker Group requires group_id and event_id.")
    if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 1:
        raise ValueError("Worker Group step_id must be a positive integer.")
    if join_policy != "ALL_TERMINAL":
        raise ValueError("Worker Group join_policy must be ALL_TERMINAL.")
    if review_status not in {"WAITING", "JOIN_READY", "REVIEWING", "REPORTED"}:
        raise ValueError("Worker Group review_status is invalid.")
    if not created_at or not updated_at:
        raise ValueError("Worker Group requires created_at and updated_at.")
    if not isinstance(slots, (list, tuple)) or not 1 <= len(slots) <= 3:
        raise ValueError("Worker Group requires between one and three slots.")

    normalized.update(
        group_id=group_id,
        event_id=event_id,
        step_id=step_id,
        join_policy=join_policy,
        review_status=review_status,
        created_at=created_at,
        updated_at=updated_at,
        slots=list(slots),
    )
    # Fail here rather than after BEGIN IMMEDIATE if the snapshot is not JSON.
    _canonical_json(normalized)
    return normalized


def _stored_worker_group(row: aiosqlite.Row) -> StoredWorkerGroup:
    raw_report = row["report_json"]
    return StoredWorkerGroup(
        snapshot=json.loads(str(row["snapshot_json"])),
        revision=int(row["revision"]),
        review_id=(None if row["review_id"] is None else str(row["review_id"])),
        report=(None if raw_report is None else json.loads(str(raw_report))),
    )


def _normalize_worker_attempt_audit(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(record)
    required_text = (
        "audit_id",
        "group_id",
        "assignment_key",
        "worker_id",
        "workspace_id",
        "checkpoint_thread_id",
        "operation",
        "outcome",
        "occurred_at",
        "recorded_at",
        "summary",
    )
    for field_name in required_text:
        value = str(normalized.get(field_name) or "").strip()
        if not value:
            raise ValueError(f"Worker audit requires {field_name}.")
        normalized[field_name] = value
    attempt_no = normalized.get("attempt_no")
    if (
        isinstance(attempt_no, bool)
        or not isinstance(attempt_no, int)
        or attempt_no < 1
    ):
        raise ValueError("Worker audit attempt_no must be positive.")
    if normalized["operation"] not in {
        "START", "FINISH", "REPLACE", "CALLBACK", "ARTIFACT_PUBLISH"
    }:
        raise ValueError("Worker audit operation is invalid.")
    if normalized["outcome"] not in {
        "APPLIED", "IDEMPOTENT", "STALE", "REJECTED"
    }:
        raise ValueError("Worker audit outcome is invalid.")
    _canonical_json(normalized)
    return normalized


async def _insert_worker_attempt_audit(
    connection: aiosqlite.Connection,
    normalized: Mapping[str, Any],
) -> bool:
    record_json = _canonical_json(normalized)
    async with connection.execute(
        """
        SELECT record_json FROM worker_attempt_audit
        WHERE audit_id = ?
        """,
        (normalized["audit_id"],),
    ) as cursor:
        existing = await cursor.fetchone()
    if existing is not None:
        if str(existing["record_json"]) != record_json:
            raise EventConflictError(
                "Worker audit_id already contains different content: "
                f"{normalized['audit_id']}"
            )
        return False
    await connection.execute(
        """
        INSERT INTO worker_attempt_audit (
            audit_id, group_id, assignment_key, attempt_no,
            worker_id, operation, outcome,
            occurred_at, recorded_at, record_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            normalized["audit_id"],
            normalized["group_id"],
            normalized["assignment_key"],
            normalized["attempt_no"],
            normalized["worker_id"],
            normalized["operation"],
            normalized["outcome"],
            normalized["occurred_at"],
            normalized["recorded_at"],
            record_json,
        ),
    )
    return True


def _event_values(event: AgentEvent) -> tuple[object, ...]:
    record = event.to_dict()
    return (
        record["event_id"],
        record["conversation_id"],
        record["action"],
        record["status"],
        record["payload_text"],
        record["target_event_id"],
        record["origin"],
        record["received_at"],
        record["status_changed_at"],
        record["reply_target_id"],
        record["result_code"],
    )


def _run_values(run: EventRun) -> tuple[object, ...]:
    record = run.to_dict()
    return (
        record["event_id"],
        record["status"],
        record["status_changed_at"],
    )


class AsyncEventStore:
    """Own durable Event/Run tables outside LangGraph's checkpoint schema."""

    def __init__(self, path: Path | str = EVENT_STORE_PATH) -> None:
        self.path = Path(path)
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncEventStore":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def start(self) -> None:
        """Open SQLite and create the Event Store schema once."""

        async with self._lock:
            if self._connection is not None:
                return

            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(str(self.path))
            connection.row_factory = aiosqlite.Row

            try:
                await connection.execute("PRAGMA foreign_keys = ON")
                await connection.execute("PRAGMA journal_mode = WAL")
                await connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        action TEXT NOT NULL CHECK (
                            action IN ('QUEUE', 'INSERT', 'REPLACE', 'CANCEL')
                        ),
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'PENDING', 'HANDLING', 'APPLIED', 'FAILED'
                            )
                        ),
                        payload_text TEXT NOT NULL,
                        target_event_id TEXT,
                        origin TEXT NOT NULL CHECK (
                            origin IN ('FEISHU', 'DESKTOP', 'SYSTEM')
                        ),
                        received_at TEXT NOT NULL,
                        status_changed_at TEXT NOT NULL,
                        reply_target_id TEXT,
                        result_code TEXT,
                        FOREIGN KEY (target_event_id)
                            REFERENCES events(event_id)
                    );

                    CREATE TABLE IF NOT EXISTS event_runs (
                        event_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'QUEUED', 'RUNNING', 'PAUSED', 'COMPLETED',
                                'CANCELLED', 'SUPERSEDED', 'FAILED'
                            )
                        ),
                        status_changed_at TEXT NOT NULL,
                        FOREIGN KEY (event_id)
                            REFERENCES events(event_id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_pending_order
                        ON events(status, received_at, event_id);

                    CREATE INDEX IF NOT EXISTS idx_events_conversation_status
                        ON events(conversation_id, status, received_at);

                    CREATE INDEX IF NOT EXISTS idx_event_runs_status
                        ON event_runs(status, status_changed_at);

                    CREATE TABLE IF NOT EXISTS worker_progress_inbox (
                        worker_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL CHECK (sequence >= 1),
                        event_id TEXT NOT NULL,
                        step_id TEXT,
                        total_tool_calls INTEGER NOT NULL CHECK (
                            total_tool_calls >= 1
                        ),
                        published_at TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        PRIMARY KEY (worker_id, sequence)
                    );

                    CREATE INDEX IF NOT EXISTS idx_worker_progress_event
                        ON worker_progress_inbox(
                            event_id, published_at, worker_id, sequence
                        );

                    CREATE TABLE IF NOT EXISTS worker_progress_cursors (
                        consumer_id TEXT NOT NULL,
                        worker_id TEXT NOT NULL,
                        last_sequence INTEGER NOT NULL CHECK (
                            last_sequence >= 0
                        ),
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (consumer_id, worker_id)
                    );

                    CREATE TABLE IF NOT EXISTS leadership_wakes (
                        wake_id TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_leadership_wakes_event
                        ON leadership_wakes(event_id, created_at, wake_id);

                    CREATE TABLE IF NOT EXISTS worker_leadership_directives (
                        wake_id TEXT NOT NULL,
                        worker_id TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('PENDING', 'APPLIED')
                        ),
                        directive_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        applied_at TEXT,
                        PRIMARY KEY (wake_id, worker_id),
                        FOREIGN KEY (wake_id)
                            REFERENCES leadership_wakes(wake_id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_worker_directives_pending
                        ON worker_leadership_directives(
                            worker_id, status, created_at, wake_id
                        );

                    CREATE TABLE IF NOT EXISTS worker_groups (
                        group_id TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL,
                        step_id INTEGER NOT NULL CHECK (step_id >= 1),
                        join_policy TEXT NOT NULL CHECK (
                            join_policy = 'ALL_TERMINAL'
                        ),
                        review_status TEXT NOT NULL CHECK (
                            review_status IN (
                                'WAITING', 'JOIN_READY',
                                'REVIEWING', 'REPORTED'
                            )
                        ),
                        revision INTEGER NOT NULL CHECK (revision >= 1),
                        review_id TEXT UNIQUE,
                        snapshot_json TEXT NOT NULL,
                        report_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_worker_groups_event_step
                        ON worker_groups(event_id, step_id, group_id);

                    CREATE INDEX IF NOT EXISTS idx_worker_groups_review
                        ON worker_groups(review_status, updated_at, group_id);

                    CREATE TABLE IF NOT EXISTS worker_attempt_audit (
                        audit_id TEXT PRIMARY KEY,
                        group_id TEXT NOT NULL,
                        assignment_key TEXT NOT NULL,
                        attempt_no INTEGER NOT NULL CHECK (attempt_no >= 1),
                        worker_id TEXT NOT NULL,
                        operation TEXT NOT NULL CHECK (
                            operation IN (
                                'START', 'FINISH', 'REPLACE', 'CALLBACK',
                                'ARTIFACT_PUBLISH'
                            )
                        ),
                        outcome TEXT NOT NULL CHECK (
                            outcome IN (
                                'APPLIED', 'IDEMPOTENT', 'STALE', 'REJECTED'
                            )
                        ),
                        occurred_at TEXT NOT NULL,
                        recorded_at TEXT NOT NULL,
                        record_json TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_worker_attempt_audit_group
                        ON worker_attempt_audit(
                            group_id, assignment_key, attempt_no,
                            occurred_at, audit_id
                        );

                    CREATE TABLE IF NOT EXISTS workspace_promotions (
                        promotion_id TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'AWAITING_APPROVAL', 'APPROVED', 'PROMOTING',
                                'DELIVERED', 'REJECTED', 'SUPERSEDED', 'FAILED'
                            )
                        ),
                        approval_mode TEXT NOT NULL CHECK (
                            approval_mode IN ('human', 'auto')
                        ),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        record_json TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_workspace_promotions_conversation
                        ON workspace_promotions(
                            conversation_id, status, created_at, promotion_id
                        );

                    CREATE INDEX IF NOT EXISTS idx_workspace_promotions_run
                        ON workspace_promotions(run_id, status, promotion_id);
                    """
                )
                # Existing local Event Stores predate durable channel reply
                # targets. SQLite's additive migration keeps their evidence
                # intact while making future interrupted Feishu runs deliverable.
                async with connection.execute(
                    "PRAGMA table_info(events)"
                ) as cursor:
                    event_columns = {
                        str(row[1]) for row in await cursor.fetchall()
                    }
                if "reply_target_id" not in event_columns:
                    await connection.execute(
                        "ALTER TABLE events ADD COLUMN reply_target_id TEXT"
                    )
                await connection.commit()
            except BaseException:
                await connection.close()
                raise

            self._connection = connection

    async def close(self) -> None:
        """Commit completed work and close the SQLite connection."""

        async with self._lock:
            connection = self._connection
            self._connection = None
            if connection is None:
                return
            await connection.commit()
            await connection.close()

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise EventStoreNotStartedError("Event Store has not been started.")
        return self._connection

    async def add_event(
        self,
        event: AgentEvent,
        *,
        run: EventRun | None = None,
    ) -> None:
        """Persist an event and its optional initial run atomically."""

        if run is not None and run.event_id != event.event_id:
            raise ValueError("EventRun must use the same event_id as its event.")

        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                if event.action in {EventAction.CANCEL, EventAction.REPLACE}:
                    async with connection.execute(
                        """
                        SELECT event_id FROM events
                        WHERE action IN ('CANCEL', 'REPLACE')
                          AND target_event_id = ?
                          AND status IN ('PENDING', 'HANDLING')
                        LIMIT 1
                        """,
                        (event.target_event_id,),
                    ) as cursor:
                        pending_cancel = await cursor.fetchone()
                    if pending_cancel is not None:
                        raise EventConflictError(
                            "Target already has a pending terminal control: "
                            f"{event.target_event_id}"
                        )
                cursor = await connection.execute(
                    """
                    INSERT INTO events (
                        event_id, conversation_id, action, status,
                        payload_text, target_event_id, origin,
                        received_at, status_changed_at, reply_target_id,
                        result_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _event_values(event),
                )
                if run is not None:
                    await connection.execute(
                        """
                        INSERT INTO event_runs (
                            event_id, status, status_changed_at
                        ) VALUES (?, ?, ?)
                        """,
                        _run_values(run),
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def add_run(self, run: EventRun) -> None:
        """Attach an execution record to an already persisted event."""

        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await connection.execute(
                    """
                    INSERT INTO event_runs (
                        event_id, status, status_changed_at
                    ) VALUES (?, ?, ?)
                    """,
                    _run_values(run),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def get_event(self, event_id: str) -> AgentEvent | None:
        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return None if row is None else AgentEvent.from_dict(dict(row))

    async def require_event(self, event_id: str) -> AgentEvent:
        event = await self.get_event(event_id)
        if event is None:
            raise EventNotFoundError(f"Unknown event_id: {event_id}")
        return event

    async def get_run(self, event_id: str) -> EventRun | None:
        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                "SELECT * FROM event_runs WHERE event_id = ?",
                (event_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return None if row is None else EventRun.from_dict(dict(row))

    async def require_run(self, event_id: str) -> EventRun:
        run = await self.get_run(event_id)
        if run is None:
            raise EventRunNotFoundError(f"Event has no run: {event_id}")
        return run

    async def recover_interrupted_runs(self) -> tuple[str, ...]:
        """Return orphaned HANDLING events to the durable input queue.

        A process can disappear after an Event has been claimed but before its
        LangGraph run completes.  ``PAUSED`` is deliberately retained as the
        resume marker; an orphaned ``RUNNING`` run is first converted to
        ``PAUSED`` because only the last synchronously committed graph node may
        be replayed safely.  A claimed-but-not-started ``QUEUED`` run is simply
        returned to the queue as a fresh execution.

        The Event and Run updates share one SQLite transaction so startup can
        never expose a requeued Event with an ambiguous Run state.
        """

        connection = self._require_connection()
        recovered_event_ids: list[str] = []
        changed_at = _utc_now()
        changed_at_text = _timestamp_text(changed_at)
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT events.*, event_runs.status AS run_status,
                           event_runs.status_changed_at AS run_status_changed_at
                    FROM events
                    INNER JOIN event_runs
                        ON event_runs.event_id = events.event_id
                    WHERE events.status = 'HANDLING'
                      AND event_runs.status IN ('QUEUED', 'RUNNING', 'PAUSED')
                    ORDER BY events.received_at, events.event_id
                    """
                ) as cursor:
                    rows = await cursor.fetchall()

                for row in rows:
                    event_values = {
                        key: row[key]
                        for key in (
                            "event_id",
                            "conversation_id",
                            "action",
                            "status",
                            "payload_text",
                            "target_event_id",
                            "origin",
                            "received_at",
                            "status_changed_at",
                            "reply_target_id",
                            "result_code",
                        )
                    }
                    current_event = AgentEvent.from_dict(event_values)
                    run_status = RunStatus(str(row["run_status"]))
                    current_run = EventRun.from_dict(
                        {
                            "event_id": current_event.event_id,
                            "status": run_status.value,
                            "status_changed_at": row["run_status_changed_at"],
                        }
                    )

                    if run_status is RunStatus.RUNNING:
                        transition_run_status(
                            current_run,
                            RunStatus.PAUSED,
                            changed_at=changed_at,
                        )
                        run_cursor = await connection.execute(
                            """
                            UPDATE event_runs
                            SET status = 'PAUSED', status_changed_at = ?
                            WHERE event_id = ? AND status = 'RUNNING'
                            """,
                            (changed_at_text, current_event.event_id),
                        )
                        if run_cursor.rowcount != 1:
                            raise EventConflictError(
                                "Run changed during startup recovery: "
                                f"{current_event.event_id}"
                            )

                    result_code = (
                        "STARTUP_REQUEUED"
                        if run_status is RunStatus.QUEUED
                        else "STARTUP_RESUME_REQUIRED"
                    )
                    transition_event_status(
                        current_event,
                        EventStatus.PENDING,
                        changed_at=changed_at,
                        result_code=result_code,
                    )
                    event_cursor = await connection.execute(
                        """
                        UPDATE events
                        SET status = 'PENDING', status_changed_at = ?,
                            result_code = ?
                        WHERE event_id = ? AND status = 'HANDLING'
                        """,
                        (
                            changed_at_text,
                            result_code,
                            current_event.event_id,
                        ),
                    )
                    if event_cursor.rowcount != 1:
                        raise EventConflictError(
                            "Event changed during startup recovery: "
                            f"{current_event.event_id}"
                        )
                    recovered_event_ids.append(current_event.event_id)

                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return tuple(recovered_event_ids)

    async def update_event_status(
        self,
        event_id: str,
        new_status: EventStatus,
        *,
        changed_at: datetime | None = None,
        result_code: str | None = None,
    ) -> AgentEvent:
        """Validate and persist one event transition atomically."""

        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._get_event_in_transaction(
                    connection,
                    event_id,
                )
                updated = transition_event_status(
                    current,
                    new_status,
                    changed_at=changed_at or _utc_now(),
                    result_code=result_code,
                )
                cursor = await connection.execute(
                    """
                    UPDATE events
                    SET status = ?, status_changed_at = ?, result_code = ?
                    WHERE event_id = ? AND status = ?
                    """,
                    (
                        updated.status.value,
                        updated.to_dict()["status_changed_at"],
                        updated.result_code,
                        event_id,
                        current.status.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise EventConflictError(
                        f"Event changed during transition: {event_id}"
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return updated

    async def update_run_status(
        self,
        event_id: str,
        new_status: RunStatus,
        *,
        changed_at: datetime | None = None,
    ) -> EventRun:
        """Validate and persist one run transition atomically."""

        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._get_run_in_transaction(
                    connection,
                    event_id,
                )
                updated = transition_run_status(
                    current,
                    new_status,
                    changed_at=changed_at or _utc_now(),
                )
                cursor = await connection.execute(
                    """
                    UPDATE event_runs
                    SET status = ?, status_changed_at = ?
                    WHERE event_id = ? AND status = ?
                    """,
                    (
                        updated.status.value,
                        updated.to_dict()["status_changed_at"],
                        event_id,
                        current.status.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise EventConflictError(
                        f"Run changed during transition: {event_id}"
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return updated

    async def apply_cancel(
        self,
        cancel_event_id: str,
        *,
        changed_at: datetime | None = None,
    ) -> CancelApplication:
        """Atomically apply a persisted CANCEL command to its target.

        CANCEL owns no LangGraph run.  Its durable effect is one transaction
        that closes the target run, closes the target task event, and records
        that the control event itself was applied.  If the target already
        reached a terminal state, the command is still consumed as an
        idempotent no-op rather than being retried forever.
        """

        normalized_cancel_id = str(cancel_event_id).strip()
        if not normalized_cancel_id:
            raise ValueError("cancel_event_id cannot be empty")
        connection = self._require_connection()
        timestamp = changed_at or _utc_now()
        timestamp_text = _timestamp_text(timestamp)

        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                command = await self._get_event_in_transaction(
                    connection,
                    normalized_cancel_id,
                )
                if command.action is not EventAction.CANCEL:
                    raise ValueError("apply_cancel requires a CANCEL event")
                if not command.target_event_id:
                    raise ValueError("CANCEL event requires target_event_id")

                target_event = await self._get_event_in_transaction(
                    connection,
                    command.target_event_id,
                )
                target_run = await self._get_run_in_transaction(
                    connection,
                    command.target_event_id,
                )
                if target_event.conversation_id != command.conversation_id:
                    raise ValueError(
                        "CANCEL cannot target an event from another conversation"
                    )

                if command.status is EventStatus.APPLIED:
                    await connection.commit()
                    return CancelApplication(
                        command_event=command,
                        target_event=target_event,
                        target_run=target_run,
                        target_was_cancelled=(
                            target_run.status is RunStatus.CANCELLED
                        ),
                    )
                if command.status not in {
                    EventStatus.PENDING,
                    EventStatus.HANDLING,
                }:
                    raise EventConflictError(
                        "CANCEL command is already terminal: "
                        f"{command.event_id}={command.status.value}"
                    )

                target_was_cancelled = target_run.status in {
                    RunStatus.QUEUED,
                    RunStatus.RUNNING,
                    RunStatus.PAUSED,
                }
                if target_was_cancelled:
                    updated_run = transition_run_status(
                        target_run,
                        RunStatus.CANCELLED,
                        changed_at=timestamp,
                    )
                    cursor = await connection.execute(
                        """
                        UPDATE event_runs
                        SET status = ?, status_changed_at = ?
                        WHERE event_id = ? AND status = ?
                        """,
                        (
                            updated_run.status.value,
                            timestamp_text,
                            target_run.event_id,
                            target_run.status.value,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise EventConflictError(
                            "Target run changed during CANCEL: "
                            f"{target_run.event_id}"
                        )
                    target_run = updated_run

                    if target_event.status in {
                        EventStatus.PENDING,
                        EventStatus.HANDLING,
                    }:
                        current_target = target_event
                        if current_target.status is EventStatus.PENDING:
                            current_target = transition_event_status(
                                current_target,
                                EventStatus.HANDLING,
                                changed_at=timestamp,
                            )
                        updated_target = transition_event_status(
                            current_target,
                            EventStatus.APPLIED,
                            changed_at=timestamp,
                            result_code="RUN_CANCELLED_BY_USER",
                        )
                        cursor = await connection.execute(
                            """
                            UPDATE events
                            SET status = ?, status_changed_at = ?, result_code = ?
                            WHERE event_id = ? AND status = ?
                            """,
                            (
                                updated_target.status.value,
                                timestamp_text,
                                updated_target.result_code,
                                target_event.event_id,
                                target_event.status.value,
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise EventConflictError(
                                "Target event changed during CANCEL: "
                                f"{target_event.event_id}"
                            )
                        target_event = updated_target

                current_command = command
                if current_command.status is EventStatus.PENDING:
                    current_command = transition_event_status(
                        current_command,
                        EventStatus.HANDLING,
                        changed_at=timestamp,
                    )
                result_code = (
                    "TARGET_CANCELLED"
                    if target_was_cancelled
                    else f"TARGET_ALREADY_{target_run.status.value}"
                )
                updated_command = transition_event_status(
                    current_command,
                    EventStatus.APPLIED,
                    changed_at=timestamp,
                    result_code=result_code,
                )
                cursor = await connection.execute(
                    """
                    UPDATE events
                    SET status = ?, status_changed_at = ?, result_code = ?
                    WHERE event_id = ? AND status = ?
                    """,
                    (
                        updated_command.status.value,
                        timestamp_text,
                        updated_command.result_code,
                        command.event_id,
                        command.status.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise EventConflictError(
                        "CANCEL command changed during application: "
                        f"{command.event_id}"
                    )

                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

        return CancelApplication(
            command_event=updated_command,
            target_event=target_event,
            target_run=target_run,
            target_was_cancelled=target_was_cancelled,
        )

    async def apply_replace(
        self,
        replacement_event_id: str,
        *,
        changed_at: datetime | None = None,
    ) -> ReplaceApplication:
        """Atomically supersede one run and claim its replacement run.

        REPLACE is both a control command and a task-producing Event.  Unlike
        CANCEL, its Event therefore remains HANDLING and its own Run remains
        QUEUED after this transaction; the run pump starts that new execution
        only after the old handler has unwound and its resources are archived.
        """

        normalized_event_id = str(replacement_event_id).strip()
        if not normalized_event_id:
            raise ValueError("replacement_event_id cannot be empty")
        connection = self._require_connection()
        timestamp = changed_at or _utc_now()
        timestamp_text = _timestamp_text(timestamp)

        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                replacement = await self._get_event_in_transaction(
                    connection,
                    normalized_event_id,
                )
                if replacement.action is not EventAction.REPLACE:
                    raise ValueError("apply_replace requires a REPLACE event")
                if not replacement.target_event_id:
                    raise ValueError("REPLACE event requires target_event_id")
                replacement_run = await self._get_run_in_transaction(
                    connection,
                    replacement.event_id,
                )
                target_event = await self._get_event_in_transaction(
                    connection,
                    replacement.target_event_id,
                )
                target_run = await self._get_run_in_transaction(
                    connection,
                    replacement.target_event_id,
                )
                if target_event.conversation_id != replacement.conversation_id:
                    raise ValueError(
                        "REPLACE cannot target an event from another conversation"
                    )
                if replacement.status not in {
                    EventStatus.PENDING,
                    EventStatus.HANDLING,
                }:
                    raise EventConflictError(
                        "REPLACE event is already terminal: "
                        f"{replacement.event_id}={replacement.status.value}"
                    )
                if replacement_run.status is not RunStatus.QUEUED:
                    raise EventConflictError(
                        "REPLACE run must still be QUEUED when it takes over: "
                        f"{replacement.event_id}={replacement_run.status.value}"
                    )

                target_was_superseded = target_run.status in {
                    RunStatus.QUEUED,
                    RunStatus.RUNNING,
                    RunStatus.PAUSED,
                }
                if target_was_superseded:
                    updated_target_run = transition_run_status(
                        target_run,
                        RunStatus.SUPERSEDED,
                        changed_at=timestamp,
                    )
                    cursor = await connection.execute(
                        """
                        UPDATE event_runs
                        SET status = ?, status_changed_at = ?
                        WHERE event_id = ? AND status = ?
                        """,
                        (
                            updated_target_run.status.value,
                            timestamp_text,
                            target_run.event_id,
                            target_run.status.value,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise EventConflictError(
                            "Target run changed during REPLACE: "
                            f"{target_run.event_id}"
                        )
                    target_run = updated_target_run

                    if target_event.status in {
                        EventStatus.PENDING,
                        EventStatus.HANDLING,
                    }:
                        current_target = target_event
                        if current_target.status is EventStatus.PENDING:
                            current_target = transition_event_status(
                                current_target,
                                EventStatus.HANDLING,
                                changed_at=timestamp,
                            )
                        updated_target_event = transition_event_status(
                            current_target,
                            EventStatus.APPLIED,
                            changed_at=timestamp,
                            result_code="RUN_SUPERSEDED_BY_USER",
                        )
                        cursor = await connection.execute(
                            """
                            UPDATE events
                            SET status = ?, status_changed_at = ?, result_code = ?
                            WHERE event_id = ? AND status = ?
                            """,
                            (
                                updated_target_event.status.value,
                                timestamp_text,
                                updated_target_event.result_code,
                                target_event.event_id,
                                target_event.status.value,
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise EventConflictError(
                                "Target event changed during REPLACE: "
                                f"{target_event.event_id}"
                            )
                        target_event = updated_target_event

                if replacement.status is EventStatus.PENDING:
                    updated_replacement = transition_event_status(
                        replacement,
                        EventStatus.HANDLING,
                        changed_at=timestamp,
                        result_code=(
                            "TARGET_SUPERSEDED"
                            if target_was_superseded
                            else f"TARGET_ALREADY_{target_run.status.value}"
                        ),
                    )
                    cursor = await connection.execute(
                        """
                        UPDATE events
                        SET status = ?, status_changed_at = ?, result_code = ?
                        WHERE event_id = ? AND status = 'PENDING'
                        """,
                        (
                            updated_replacement.status.value,
                            timestamp_text,
                            updated_replacement.result_code,
                            replacement.event_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise EventConflictError(
                            "REPLACE event changed during application: "
                            f"{replacement.event_id}"
                        )
                    replacement = updated_replacement

                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

        return ReplaceApplication(
            replacement_event=replacement,
            replacement_run=replacement_run,
            target_event=target_event,
            target_run=target_run,
            target_was_superseded=target_was_superseded,
        )

    async def claim_next_pending(
        self,
        *,
        conversation_id: str | None = None,
        changed_at: datetime | None = None,
    ) -> AgentEvent | None:
        """Atomically claim the oldest pending event for one worker."""

        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                if conversation_id is None:
                    query = (
                        "SELECT * FROM events WHERE status = 'PENDING' "
                        "ORDER BY CASE action "
                        "WHEN 'CANCEL' THEN 0 WHEN 'REPLACE' THEN 1 "
                        "WHEN 'INSERT' THEN 2 ELSE 3 END, "
                        "received_at, event_id LIMIT 1"
                    )
                    parameters: tuple[object, ...] = ()
                else:
                    query = (
                        "SELECT * FROM events "
                        "WHERE status = 'PENDING' AND conversation_id = ? "
                        "ORDER BY CASE action "
                        "WHEN 'CANCEL' THEN 0 WHEN 'REPLACE' THEN 1 "
                        "WHEN 'INSERT' THEN 2 ELSE 3 END, "
                        "received_at, event_id LIMIT 1"
                    )
                    parameters = (conversation_id,)

                async with connection.execute(query, parameters) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    await connection.commit()
                    return None

                current = AgentEvent.from_dict(dict(row))
                claimed = transition_event_status(
                    current,
                    EventStatus.HANDLING,
                    changed_at=changed_at or _utc_now(),
                )
                cursor = await connection.execute(
                    """
                    UPDATE events
                    SET status = ?, status_changed_at = ?, result_code = NULL
                    WHERE event_id = ? AND status = 'PENDING'
                    """,
                    (
                        claimed.status.value,
                        claimed.to_dict()["status_changed_at"],
                        claimed.event_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise EventConflictError(
                        f"Event was claimed concurrently: {claimed.event_id}"
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return claimed

    async def claim_pending_insert(
        self,
        *,
        target_event_id: str,
        changed_at: datetime | None = None,
    ) -> AgentEvent | None:
        """Claim the oldest INSERT that explicitly targets a paused run."""

        normalized_target = str(target_event_id).strip()
        if not normalized_target:
            raise ValueError("target_event_id cannot be empty")
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT * FROM events
                    WHERE status = 'PENDING'
                      AND action = 'INSERT'
                      AND target_event_id = ?
                    ORDER BY received_at, event_id
                    LIMIT 1
                    """,
                    (normalized_target,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    await connection.commit()
                    return None
                current = AgentEvent.from_dict(dict(row))
                claimed = transition_event_status(
                    current,
                    EventStatus.HANDLING,
                    changed_at=changed_at or _utc_now(),
                )
                cursor = await connection.execute(
                    """
                    UPDATE events
                    SET status = ?, status_changed_at = ?, result_code = NULL
                    WHERE event_id = ? AND status = 'PENDING'
                    """,
                    (
                        claimed.status.value,
                        claimed.to_dict()["status_changed_at"],
                        claimed.event_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise EventConflictError(
                        f"INSERT changed during claim: {claimed.event_id}"
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return claimed

    async def list_events(
        self,
        *,
        statuses: Iterable[EventStatus] | None = None,
        conversation_id: str | None = None,
    ) -> list[AgentEvent]:
        """List events in deterministic receive order."""

        connection = self._require_connection()
        clauses: list[str] = []
        parameters: list[object] = []

        if statuses is not None:
            status_values = [status.value for status in statuses]
            if not status_values:
                return []
            placeholders = ", ".join("?" for _ in status_values)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(status_values)
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            parameters.append(conversation_id)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self._lock:
            async with connection.execute(
                f"SELECT * FROM events{where} ORDER BY received_at, event_id",
                tuple(parameters),
            ) as cursor:
                rows = await cursor.fetchall()
        return [AgentEvent.from_dict(dict(row)) for row in rows]

    async def list_runs(
        self,
        *,
        statuses: Iterable[RunStatus] | None = None,
    ) -> list[EventRun]:
        """List execution records in status-change order."""

        connection = self._require_connection()
        parameters: tuple[object, ...] = ()
        where = ""
        if statuses is not None:
            status_values = [status.value for status in statuses]
            if not status_values:
                return []
            placeholders = ", ".join("?" for _ in status_values)
            where = f" WHERE status IN ({placeholders})"
            parameters = tuple(status_values)

        async with self._lock:
            async with connection.execute(
                "SELECT * FROM event_runs"
                f"{where} ORDER BY status_changed_at, event_id",
                parameters,
            ) as cursor:
                rows = await cursor.fetchall()
        return [EventRun.from_dict(dict(row)) for row in rows]

    async def append_worker_progress(
        self,
        record: Mapping[str, Any],
    ) -> bool:
        """Append one validated Worker report to the public control inbox.

        The Worker layer owns the Pydantic schema. This persistence boundary
        stores that JSON representation without importing Worker runtime code.
        Replaying the same record is idempotent; reusing the same
        ``(worker_id, sequence)`` for different content is a conflict.
        """

        normalized = dict(record)
        worker_id = str(normalized.get("worker_id") or "").strip()
        event_id = str(normalized.get("event_id") or "").strip()
        raw_step_id = normalized.get("step_id")
        step_id = None if raw_step_id is None else str(raw_step_id).strip()
        sequence = normalized.get("sequence")
        total_tool_calls = normalized.get("total_tool_calls")
        published_at = str(normalized.get("published_at") or "").strip()

        if not worker_id:
            raise ValueError("Worker progress requires worker_id.")
        if not event_id:
            raise ValueError("Worker progress requires event_id.")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("Worker progress sequence must be a positive integer.")
        if (
            isinstance(total_tool_calls, bool)
            or not isinstance(total_tool_calls, int)
            or total_tool_calls < 1
        ):
            raise ValueError(
                "Worker progress total_tool_calls must be a positive integer."
            )
        if not published_at:
            raise ValueError("Worker progress requires published_at.")
        if not isinstance(normalized.get("progress"), dict):
            raise ValueError("Worker progress requires a progress object.")

        normalized.update(
            {
                "worker_id": worker_id,
                "event_id": event_id,
                "step_id": step_id or None,
                "sequence": sequence,
                "total_tool_calls": total_tool_calls,
                "published_at": published_at,
            }
        )
        record_json = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT record_json
                    FROM worker_progress_inbox
                    WHERE worker_id = ? AND sequence = ?
                    """,
                    (worker_id, sequence),
                ) as cursor:
                    existing = await cursor.fetchone()

                if existing is not None:
                    existing_record = json.loads(str(existing["record_json"]))
                    replay_record = dict(normalized)
                    existing_record.pop("published_at", None)
                    replay_record.pop("published_at", None)
                    if existing_record != replay_record:
                        raise EventConflictError(
                            "Worker progress sequence already contains "
                            f"different content: {worker_id}#{sequence}"
                        )
                    await connection.commit()
                    return False

                await connection.execute(
                    """
                    INSERT INTO worker_progress_inbox (
                        worker_id, sequence, event_id, step_id,
                        total_tool_calls, published_at, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        worker_id,
                        sequence,
                        event_id,
                        step_id or None,
                        total_tool_calls,
                        published_at,
                        record_json,
                    ),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return True

    async def list_worker_progress(
        self,
        *,
        worker_id: str,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read one Worker''s append-only reports after a leadership cursor."""

        normalized_worker_id = worker_id.strip()
        if not normalized_worker_id:
            raise ValueError("worker_id cannot be empty.")
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative.")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000.")

        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                """
                SELECT record_json
                FROM worker_progress_inbox
                WHERE worker_id = ? AND sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (normalized_worker_id, after_sequence, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [json.loads(str(row["record_json"])) for row in rows]

    async def list_event_worker_progress(
        self,
        event_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read reports across all Workers owned by one event."""

        normalized_event_id = event_id.strip()
        if not normalized_event_id:
            raise ValueError("event_id cannot be empty.")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000.")

        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                """
                SELECT record_json
                FROM worker_progress_inbox
                WHERE event_id = ?
                ORDER BY published_at, worker_id, sequence
                LIMIT ?
                """,
                (normalized_event_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [json.loads(str(row["record_json"])) for row in rows]

    async def get_worker_progress_cursor(
        self,
        *,
        consumer_id: str,
        worker_id: str,
    ) -> int:
        """Return the last report sequence acknowledged by one leader."""

        normalized_consumer_id = consumer_id.strip()
        normalized_worker_id = worker_id.strip()
        if not normalized_consumer_id or not normalized_worker_id:
            raise ValueError("consumer_id and worker_id cannot be empty.")

        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                """
                SELECT last_sequence
                FROM worker_progress_cursors
                WHERE consumer_id = ? AND worker_id = ?
                """,
                (normalized_consumer_id, normalized_worker_id),
            ) as cursor:
                row = await cursor.fetchone()
        return 0 if row is None else int(row["last_sequence"])

    async def advance_worker_progress_cursor(
        self,
        *,
        consumer_id: str,
        worker_id: str,
        sequence: int,
        changed_at: datetime | None = None,
    ) -> int:
        """Advance a leader cursor monotonically and persist the new position."""

        normalized_consumer_id = consumer_id.strip()
        normalized_worker_id = worker_id.strip()
        if not normalized_consumer_id or not normalized_worker_id:
            raise ValueError("consumer_id and worker_id cannot be empty.")
        if sequence < 0:
            raise ValueError("sequence cannot be negative.")
        timestamp = (changed_at or _utc_now()).astimezone(timezone.utc)
        timestamp_text = timestamp.isoformat().replace("+00:00", "Z")

        connection = self._require_connection()
        async with self._lock:
            await connection.execute(
                """
                INSERT INTO worker_progress_cursors (
                    consumer_id, worker_id, last_sequence, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(consumer_id, worker_id) DO UPDATE SET
                    last_sequence = MAX(last_sequence, excluded.last_sequence),
                    updated_at = CASE
                        WHEN excluded.last_sequence > last_sequence
                        THEN excluded.updated_at
                        ELSE updated_at
                    END
                """,
                (
                    normalized_consumer_id,
                    normalized_worker_id,
                    sequence,
                    timestamp_text,
                ),
            )
            await connection.commit()

        return await self.get_worker_progress_cursor(
            consumer_id=normalized_consumer_id,
            worker_id=normalized_worker_id,
        )

    async def save_leadership_wake(
        self,
        *,
        wake: Mapping[str, Any],
        result: Mapping[str, Any],
        directives: Mapping[str, Mapping[str, Any]],
        cursor_updates: Mapping[str, int],
        consumer_id: str,
    ) -> None:
        """Atomically save a decision, pending directives, and input watermarks."""

        wake_value = dict(wake)
        wake_id = str(wake_value.get("wake_id") or "").strip()
        event_id = str(wake_value.get("event_id") or "").strip()
        reason = str(wake_value.get("reason") or "").strip()
        created_at = str(wake_value.get("created_at") or "").strip()
        normalized_consumer_id = consumer_id.strip()
        if not all((wake_id, event_id, reason, created_at, normalized_consumer_id)):
            raise ValueError("Leadership wake identity fields cannot be empty.")

        request_json = json.dumps(
            wake_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        result_json = json.dumps(
            dict(result),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await connection.execute(
                    """
                    INSERT INTO leadership_wakes (
                        wake_id, event_id, reason, request_json,
                        result_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        wake_id,
                        event_id,
                        reason,
                        request_json,
                        result_json,
                        created_at,
                    ),
                )

                for worker_id, directive in directives.items():
                    normalized_worker_id = str(worker_id).strip()
                    if not normalized_worker_id:
                        raise ValueError("Directive worker_id cannot be empty.")
                    directive_json = json.dumps(
                        dict(directive),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    await connection.execute(
                        """
                        INSERT INTO worker_leadership_directives (
                            wake_id, worker_id, event_id, status,
                            directive_json, created_at, applied_at
                        ) VALUES (?, ?, ?, 'PENDING', ?, ?, NULL)
                        """,
                        (
                            wake_id,
                            normalized_worker_id,
                            event_id,
                            directive_json,
                            created_at,
                        ),
                    )

                for worker_id, sequence in cursor_updates.items():
                    normalized_worker_id = str(worker_id).strip()
                    if not normalized_worker_id or sequence < 0:
                        raise ValueError("Invalid leadership cursor update.")
                    await connection.execute(
                        """
                        INSERT INTO worker_progress_cursors (
                            consumer_id, worker_id, last_sequence, updated_at
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(consumer_id, worker_id) DO UPDATE SET
                            last_sequence = MAX(
                                last_sequence, excluded.last_sequence
                            ),
                            updated_at = CASE
                                WHEN excluded.last_sequence > last_sequence
                                THEN excluded.updated_at
                                ELSE updated_at
                            END
                        """,
                        (
                            normalized_consumer_id,
                            normalized_worker_id,
                            sequence,
                            created_at,
                        ),
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def get_pending_worker_directive(
        self,
        worker_id: str,
    ) -> dict[str, Any] | None:
        """Return the oldest durable directive not yet applied by a Worker."""

        normalized_worker_id = worker_id.strip()
        if not normalized_worker_id:
            raise ValueError("worker_id cannot be empty.")
        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                """
                SELECT wake_id, directive_json
                FROM worker_leadership_directives
                WHERE worker_id = ? AND status = 'PENDING'
                ORDER BY created_at, wake_id
                LIMIT 1
                """,
                (normalized_worker_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "wake_id": str(row["wake_id"]),
            "directive": json.loads(str(row["directive_json"])),
        }

    async def mark_worker_directive_applied(
        self,
        *,
        wake_id: str,
        worker_id: str,
        changed_at: datetime | None = None,
    ) -> None:
        """Mark delivery only after the resumed Worker checkpoint is observed."""

        normalized_wake_id = wake_id.strip()
        normalized_worker_id = worker_id.strip()
        if not normalized_wake_id or not normalized_worker_id:
            raise ValueError("wake_id and worker_id cannot be empty.")
        timestamp = (changed_at or _utc_now()).astimezone(timezone.utc)
        timestamp_text = timestamp.isoformat().replace("+00:00", "Z")
        connection = self._require_connection()
        async with self._lock:
            cursor = await connection.execute(
                """
                UPDATE worker_leadership_directives
                SET status = 'APPLIED', applied_at = ?
                WHERE wake_id = ? AND worker_id = ? AND status = 'PENDING'
                """,
                (timestamp_text, normalized_wake_id, normalized_worker_id),
            )
            if cursor.rowcount not in {0, 1}:
                raise EventConflictError("Directive apply updated multiple rows.")
            await connection.commit()

    async def create_worker_group(
        self,
        snapshot: Mapping[str, Any],
    ) -> StoredWorkerGroup:
        """Persist the first immutable snapshot; exact replay is idempotent."""

        normalized = _normalize_worker_group_snapshot(snapshot)
        if normalized["review_status"] in {"REVIEWING", "REPORTED"}:
            raise ValueError(
                "A new Worker Group cannot begin in a claimed Reporter state."
            )
        snapshot_json = _canonical_json(normalized)
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    "SELECT * FROM worker_groups WHERE group_id = ?",
                    (normalized["group_id"],),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None:
                    record = _stored_worker_group(existing)
                    if _canonical_json(record.snapshot) != snapshot_json:
                        raise EventConflictError(
                            "Worker Group already exists with different content: "
                            f"{normalized['group_id']}"
                        )
                    await connection.commit()
                    return record

                await connection.execute(
                    """
                    INSERT INTO worker_groups (
                        group_id, event_id, step_id, join_policy,
                        review_status, revision, review_id,
                        snapshot_json, report_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, NULL, ?, NULL, ?, ?)
                    """,
                    (
                        normalized["group_id"],
                        normalized["event_id"],
                        normalized["step_id"],
                        normalized["join_policy"],
                        normalized["review_status"],
                        snapshot_json,
                        normalized["created_at"],
                        normalized["updated_at"],
                    ),
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        record = await self.require_worker_group(normalized["group_id"])
        return record

    async def get_worker_group(
        self,
        group_id: str,
    ) -> StoredWorkerGroup | None:
        """Read one group snapshot and its current optimistic revision."""

        normalized_group_id = str(group_id).strip()
        if not normalized_group_id:
            raise ValueError("group_id cannot be empty.")
        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                "SELECT * FROM worker_groups WHERE group_id = ?",
                (normalized_group_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return None if row is None else _stored_worker_group(row)

    async def require_worker_group(
        self,
        group_id: str,
    ) -> StoredWorkerGroup:
        record = await self.get_worker_group(group_id)
        if record is None:
            raise WorkerGroupNotFoundError(f"Unknown Worker Group: {group_id}")
        return record

    async def compare_and_set_worker_group(
        self,
        snapshot: Mapping[str, Any],
        *,
        expected_revision: int,
        audit_record: Mapping[str, Any] | None = None,
    ) -> WorkerGroupCASResult:
        """Write a caller-validated snapshot only if its revision is current."""

        if expected_revision < 1:
            raise ValueError("expected_revision must be positive.")
        normalized = _normalize_worker_group_snapshot(snapshot)
        normalized_audit = (
            None
            if audit_record is None
            else _normalize_worker_attempt_audit(audit_record)
        )
        if (
            normalized_audit is not None
            and normalized_audit["group_id"] != normalized["group_id"]
        ):
            raise ValueError("Worker audit and snapshot must use the same group_id.")
        if normalized["review_status"] in {"REVIEWING", "REPORTED"}:
            raise ValueError(
                "Reporter states may only change through claim/complete methods."
            )
        snapshot_json = _canonical_json(normalized)
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    "SELECT * FROM worker_groups WHERE group_id = ?",
                    (normalized["group_id"],),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise WorkerGroupNotFoundError(
                        f"Unknown Worker Group: {normalized['group_id']}"
                    )
                current = _stored_worker_group(row)
                persisted = current.snapshot
                immutable_fields = (
                    "group_id",
                    "event_id",
                    "step_id",
                    "join_policy",
                    "created_at",
                )
                if any(
                    normalized[field] != persisted[field]
                    for field in immutable_fields
                ):
                    raise EventConflictError(
                        "Worker Group immutable identity fields cannot change."
                    )
                if current.revision != expected_revision:
                    await connection.commit()
                    return WorkerGroupCASResult(applied=False, record=current)
                if str(persisted["review_status"]) in {"REVIEWING", "REPORTED"}:
                    raise EventConflictError(
                        "A claimed Worker Group cannot accept Worker snapshots."
                    )

                cursor = await connection.execute(
                    """
                    UPDATE worker_groups
                    SET review_status = ?, revision = revision + 1,
                        snapshot_json = ?, updated_at = ?
                    WHERE group_id = ? AND revision = ?
                      AND review_status IN ('WAITING', 'JOIN_READY')
                    """,
                    (
                        normalized["review_status"],
                        snapshot_json,
                        normalized["updated_at"],
                        normalized["group_id"],
                        expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise EventConflictError(
                        "Worker Group changed during its CAS transaction."
                    )
                if normalized_audit is not None:
                    await _insert_worker_attempt_audit(
                        connection,
                        normalized_audit,
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return WorkerGroupCASResult(
            applied=True,
            record=await self.require_worker_group(normalized["group_id"]),
        )

    async def claim_worker_group_review(
        self,
        *,
        group_id: str,
        review_id: str,
        changed_at: datetime | None = None,
    ) -> WorkerGroupReviewClaim:
        """Atomically grant Reporter execution to one caller only."""

        normalized_group_id = str(group_id).strip()
        normalized_review_id = str(review_id).strip()
        if not normalized_group_id or not normalized_review_id:
            raise ValueError("group_id and review_id cannot be empty.")
        timestamp = _timestamp_text(changed_at)
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    "SELECT * FROM worker_groups WHERE group_id = ?",
                    (normalized_group_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise WorkerGroupNotFoundError(
                        f"Unknown Worker Group: {normalized_group_id}"
                    )
                current = _stored_worker_group(row)
                status = str(current.snapshot["review_status"])
                if status == "JOIN_READY":
                    claimed_snapshot = dict(current.snapshot)
                    claimed_snapshot.update(
                        review_status="REVIEWING",
                        updated_at=timestamp,
                    )
                    cursor = await connection.execute(
                        """
                        UPDATE worker_groups
                        SET review_status = 'REVIEWING',
                            revision = revision + 1,
                            review_id = ?, snapshot_json = ?, updated_at = ?
                        WHERE group_id = ? AND review_status = 'JOIN_READY'
                          AND revision = ? AND review_id IS NULL
                        """,
                        (
                            normalized_review_id,
                            _canonical_json(claimed_snapshot),
                            timestamp,
                            normalized_group_id,
                            current.revision,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise EventConflictError(
                            "Worker Group review claim changed during transaction."
                        )
                    async with connection.execute(
                        "SELECT * FROM worker_groups WHERE group_id = ?",
                        (normalized_group_id,),
                    ) as claimed_cursor:
                        claimed_row = await claimed_cursor.fetchone()
                    if claimed_row is None:
                        raise WorkerGroupNotFoundError(
                            f"Unknown Worker Group: {normalized_group_id}"
                        )
                    claimed = _stored_worker_group(claimed_row)
                    await connection.commit()
                    return WorkerGroupReviewClaim(
                        acquired_now=True,
                        owned_by_caller=True,
                        record=claimed,
                    )

                await connection.commit()
                return WorkerGroupReviewClaim(
                    acquired_now=False,
                    owned_by_caller=(
                        status == "REVIEWING"
                        and current.review_id == normalized_review_id
                    ),
                    record=current,
                )
            except BaseException:
                await connection.rollback()
                raise

    async def complete_worker_group_review(
        self,
        *,
        group_id: str,
        review_id: str,
        report: Mapping[str, Any],
        changed_at: datetime | None = None,
    ) -> StoredWorkerGroup:
        """Persist one Reporter result only for the durable claim owner."""

        normalized_group_id = str(group_id).strip()
        normalized_review_id = str(review_id).strip()
        if not normalized_group_id or not normalized_review_id:
            raise ValueError("group_id and review_id cannot be empty.")
        report_json = _canonical_json(report)
        timestamp = _timestamp_text(changed_at)
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    "SELECT * FROM worker_groups WHERE group_id = ?",
                    (normalized_group_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise WorkerGroupNotFoundError(
                        f"Unknown Worker Group: {normalized_group_id}"
                    )
                current = _stored_worker_group(row)
                status = str(current.snapshot["review_status"])
                if status == "REPORTED":
                    if (
                        current.review_id == normalized_review_id
                        and current.report is not None
                        and _canonical_json(current.report) == report_json
                    ):
                        await connection.commit()
                        return current
                    raise EventConflictError(
                        "Worker Group already contains a different Reporter result."
                    )
                if status != "REVIEWING" or current.review_id != normalized_review_id:
                    raise EventConflictError(
                        "Only the current durable review owner may complete Reporter."
                    )

                reported_snapshot = dict(current.snapshot)
                reported_snapshot.update(
                    review_status="REPORTED",
                    updated_at=timestamp,
                )
                cursor = await connection.execute(
                    """
                    UPDATE worker_groups
                    SET review_status = 'REPORTED',
                        revision = revision + 1,
                        snapshot_json = ?, report_json = ?, updated_at = ?
                    WHERE group_id = ? AND review_status = 'REVIEWING'
                      AND review_id = ? AND revision = ?
                    """,
                    (
                        _canonical_json(reported_snapshot),
                        report_json,
                        timestamp,
                        normalized_group_id,
                        normalized_review_id,
                        current.revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise EventConflictError(
                        "Worker Group review completion changed during transaction."
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return await self.require_worker_group(normalized_group_id)

    async def append_worker_attempt_audit(
        self,
        record: Mapping[str, Any],
    ) -> bool:
        """Append one bounded attempt event; exact replay is idempotent."""

        normalized = _normalize_worker_attempt_audit(record)
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                inserted = await _insert_worker_attempt_audit(
                    connection,
                    normalized,
                )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return inserted

    async def list_worker_attempt_audit(
        self,
        *,
        group_id: str,
        assignment_key: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Read bounded audit envelopes without loading raw Worker traces."""

        normalized_group_id = str(group_id).strip()
        normalized_key = None if assignment_key is None else assignment_key.strip()
        if not normalized_group_id:
            raise ValueError("group_id cannot be empty.")
        if assignment_key is not None and not normalized_key:
            raise ValueError("assignment_key cannot be empty.")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000.")
        parameters: tuple[Any, ...]
        where = "group_id = ?"
        parameters = (normalized_group_id,)
        if normalized_key is not None:
            where += " AND assignment_key = ?"
            parameters = (*parameters, normalized_key)
        parameters = (*parameters, limit)
        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                f"""
                SELECT record_json FROM worker_attempt_audit
                WHERE {where}
                ORDER BY occurred_at, audit_id
                LIMIT ?
                """,
                parameters,
            ) as cursor:
                rows = await cursor.fetchall()
        return [json.loads(str(row["record_json"])) for row in rows]

    async def add_workspace_promotion(
        self,
        promotion: WorkspacePromotion,
    ) -> WorkspacePromotion:
        """Persist one immutable promotion request; exact replay is idempotent."""

        record = promotion.model_dump(mode="json")
        payload = _canonical_json(record)
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT record_json FROM workspace_promotions
                    WHERE promotion_id = ?
                    """,
                    (promotion.promotion_id,),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None:
                    stored = WorkspacePromotion.model_validate_json(
                        str(existing["record_json"])
                    )
                    if stored != promotion:
                        raise EventConflictError(
                            "promotion identity was reused for different content: "
                            f"{promotion.promotion_id}"
                        )
                    await connection.commit()
                    return stored
                await connection.execute(
                    """
                    INSERT INTO workspace_promotions (
                        promotion_id, event_id, run_id, conversation_id,
                        status, approval_mode, created_at, updated_at,
                        record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        promotion.promotion_id,
                        promotion.event_id,
                        promotion.run_id,
                        promotion.conversation_id,
                        promotion.status.value,
                        promotion.approval_mode.value,
                        promotion.created_at.isoformat(),
                        promotion.updated_at.isoformat(),
                        payload,
                    ),
                )
                await connection.commit()
                return promotion
            except BaseException:
                await connection.rollback()
                raise

    async def get_workspace_promotion(
        self,
        promotion_id: str,
    ) -> WorkspacePromotion | None:
        normalized = str(promotion_id).strip()
        if not normalized:
            raise ValueError("promotion_id cannot be empty")
        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                """
                SELECT record_json FROM workspace_promotions
                WHERE promotion_id = ?
                """,
                (normalized,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return WorkspacePromotion.model_validate_json(str(row["record_json"]))

    async def list_workspace_promotions(
        self,
        *,
        conversation_id: str | None = None,
        run_id: str | None = None,
        statuses: Iterable[PromotionStatus] | None = None,
        limit: int = 200,
    ) -> list[WorkspacePromotion]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses: list[str] = []
        parameters: list[Any] = []
        if conversation_id is not None:
            normalized_conversation = str(conversation_id).strip()
            if not normalized_conversation:
                raise ValueError("conversation_id cannot be empty")
            clauses.append("conversation_id = ?")
            parameters.append(normalized_conversation)
        if run_id is not None:
            normalized_run = str(run_id).strip()
            if not normalized_run:
                raise ValueError("run_id cannot be empty")
            clauses.append("run_id = ?")
            parameters.append(normalized_run)
        if statuses is not None:
            normalized_statuses = tuple(
                status if isinstance(status, PromotionStatus) else PromotionStatus(status)
                for status in statuses
            )
            if not normalized_statuses:
                return []
            clauses.append(
                "status IN (" + ", ".join("?" for _ in normalized_statuses) + ")"
            )
            parameters.extend(status.value for status in normalized_statuses)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(limit)
        connection = self._require_connection()
        async with self._lock:
            async with connection.execute(
                f"""
                SELECT record_json FROM workspace_promotions
                {where}
                ORDER BY created_at DESC, promotion_id DESC
                LIMIT ?
                """,
                tuple(parameters),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            WorkspacePromotion.model_validate_json(str(row["record_json"]))
            for row in rows
        ]

    async def compare_and_set_workspace_promotion(
        self,
        *,
        expected_status: PromotionStatus,
        promotion: WorkspacePromotion,
    ) -> WorkspacePromotion:
        """Atomically persist one validated promotion state transition."""

        payload = _canonical_json(promotion.model_dump(mode="json"))
        connection = self._require_connection()
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """
                    SELECT record_json FROM workspace_promotions
                    WHERE promotion_id = ?
                    """,
                    (promotion.promotion_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise EventNotFoundError(
                        f"Unknown promotion_id: {promotion.promotion_id}"
                    )
                current = WorkspacePromotion.model_validate_json(
                    str(row["record_json"])
                )
                if current == promotion:
                    await connection.commit()
                    return current
                if current.status is not expected_status:
                    raise EventConflictError(
                        "promotion changed before transition: "
                        f"expected={expected_status.value}, actual={current.status.value}"
                    )
                update_cursor = await connection.execute(
                    """
                    UPDATE workspace_promotions
                    SET status = ?, approval_mode = ?, updated_at = ?, record_json = ?
                    WHERE promotion_id = ? AND status = ?
                    """,
                    (
                        promotion.status.value,
                        promotion.approval_mode.value,
                        promotion.updated_at.isoformat(),
                        payload,
                        promotion.promotion_id,
                        expected_status.value,
                    ),
                )
                if update_cursor.rowcount != 1:
                    raise EventConflictError(
                        f"promotion transition lost race: {promotion.promotion_id}"
                    )
                await connection.commit()
                return promotion
            except BaseException:
                await connection.rollback()
                raise

    async def _get_event_in_transaction(
        self,
        connection: aiosqlite.Connection,
        event_id: str,
    ) -> AgentEvent:
        async with connection.execute(
            "SELECT * FROM events WHERE event_id = ?",
            (event_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise EventNotFoundError(f"Unknown event_id: {event_id}")
        return AgentEvent.from_dict(dict(row))

    async def _get_run_in_transaction(
        self,
        connection: aiosqlite.Connection,
        event_id: str,
    ) -> EventRun:
        async with connection.execute(
            "SELECT * FROM event_runs WHERE event_id = ?",
            (event_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise EventRunNotFoundError(f"Event has no run: {event_id}")
        return EventRun.from_dict(dict(row))
