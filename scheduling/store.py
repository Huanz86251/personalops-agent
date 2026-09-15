"""SQLite persistence and atomic claiming for scheduled reminders."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import aiosqlite

from path import AGENT_DATA_ROOT
from scheduling.models import (
    DeliveryTarget,
    Recurrence,
    ReminderRun,
    ReminderRunStatus,
    ReminderSchedule,
    ScheduleStatus,
    create_schedule,
    datetime_to_text,
    next_occurrence,
)


SCHEDULE_STORE_PATH = AGENT_DATA_ROOT / "schedules.sqlite3"


class ScheduleNotFoundError(LookupError):
    pass


def _schedule_from_row(row: aiosqlite.Row) -> ReminderSchedule:
    return ReminderSchedule.from_dict(dict(row))


def _run_from_row(row: aiosqlite.Row) -> ReminderRun:
    return ReminderRun.from_dict(dict(row))


class AsyncScheduleStore:
    def __init__(self, path: Path | str = SCHEDULE_STORE_PATH) -> None:
        self.path = Path(path)
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._connection is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(str(self.path))
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reminder_schedules (
                    schedule_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    delivery_target TEXT NOT NULL DEFAULT 'WINDOWS'
                        CHECK (delivery_target IN ('WINDOWS','FEISHU','AGENT_EVENT')),
                    reply_target_id TEXT,
                    conversation_id TEXT,
                    recurrence TEXT NOT NULL CHECK (recurrence IN ('ONCE','DAILY','WEEKLY')),
                    timezone_name TEXT NOT NULL,
                    next_run_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('ACTIVE','PAUSED','COMPLETED','CANCELLED')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reminder_schedules_due
                    ON reminder_schedules(status, next_run_at, schedule_id);
                CREATE TABLE IF NOT EXISTS reminder_runs (
                    run_id TEXT PRIMARY KEY,
                    schedule_id TEXT NOT NULL,
                    scheduled_for TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('PENDING','CLAIMED','SUBMITTED','FAILED')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    UNIQUE(schedule_id, scheduled_for),
                    FOREIGN KEY(schedule_id) REFERENCES reminder_schedules(schedule_id)
                );
                CREATE INDEX IF NOT EXISTS idx_reminder_runs_pending
                    ON reminder_runs(status, created_at, run_id);
                """
            )
            async with connection.execute(
                "PRAGMA table_info(reminder_schedules)"
            ) as cursor:
                schedule_columns = {str(row["name"]) for row in await cursor.fetchall()}
            migrations = {
                "delivery_target": (
                    "ALTER TABLE reminder_schedules ADD COLUMN delivery_target "
                    "TEXT NOT NULL DEFAULT 'WINDOWS'"
                ),
                "reply_target_id": (
                    "ALTER TABLE reminder_schedules ADD COLUMN reply_target_id TEXT"
                ),
                "conversation_id": (
                    "ALTER TABLE reminder_schedules ADD COLUMN conversation_id TEXT"
                ),
            }
            for column_name, statement in migrations.items():
                if column_name not in schedule_columns:
                    await connection.execute(statement)
            await connection.commit()
            self._connection = connection

    async def close(self) -> None:
        async with self._lock:
            connection = self._connection
            self._connection = None
            if connection is not None:
                await connection.commit()
                await connection.close()

    def _connection_or_raise(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Schedule Store尚未启动。")
        return self._connection

    async def create(
        self,
        *,
        title: str,
        message: str,
        run_at: datetime,
        timezone_name: str,
        recurrence: Recurrence,
        delivery_target: DeliveryTarget = DeliveryTarget.WINDOWS,
        reply_target_id: str | None = None,
        conversation_id: str | None = None,
        now: datetime | None = None,
    ) -> ReminderSchedule:
        schedule = create_schedule(
            title=title,
            message=message,
            run_at=run_at,
            timezone_name=timezone_name,
            recurrence=recurrence,
            delivery_target=delivery_target,
            reply_target_id=reply_target_id,
            conversation_id=conversation_id,
            now=now,
        )
        connection = self._connection_or_raise()
        record = schedule.to_dict()
        async with self._lock:
            await connection.execute(
                """INSERT INTO reminder_schedules (
                    schedule_id,title,message,delivery_target,reply_target_id,
                    conversation_id,recurrence,timezone_name,next_run_at,status,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                tuple(record[key] for key in (
                    "schedule_id", "title", "message", "delivery_target",
                    "reply_target_id", "conversation_id", "recurrence", "timezone_name",
                    "next_run_at", "status", "created_at", "updated_at",
                )),
            )
            await connection.commit()
        return schedule

    async def get(self, schedule_id: str) -> ReminderSchedule | None:
        connection = self._connection_or_raise()
        async with self._lock:
            async with connection.execute(
                "SELECT * FROM reminder_schedules WHERE schedule_id = ?",
                (str(schedule_id).strip(),),
            ) as cursor:
                row = await cursor.fetchone()
        return None if row is None else _schedule_from_row(row)

    async def list(
        self,
        statuses: Iterable[ScheduleStatus] | None = None,
        *,
        limit: int = 50,
    ) -> list[ReminderSchedule]:
        if not 1 <= limit <= 200:
            raise ValueError("limit必须在1到200之间。")
        connection = self._connection_or_raise()
        parameters: list[object] = []
        where = ""
        if statuses is not None:
            values = [item.value for item in statuses]
            if not values:
                return []
            where = f"WHERE status IN ({','.join('?' for _ in values)})"
            parameters.extend(values)
        parameters.append(limit)
        async with self._lock:
            async with connection.execute(
                f"SELECT * FROM reminder_schedules {where} ORDER BY next_run_at, schedule_id LIMIT ?",
                parameters,
            ) as cursor:
                rows = await cursor.fetchall()
        return [_schedule_from_row(row) for row in rows]

    async def set_status(self, schedule_id: str, status: ScheduleStatus) -> ReminderSchedule:
        connection = self._connection_or_raise()
        normalized_id = str(schedule_id).strip()
        now_text = datetime_to_text(datetime.now(timezone.utc))
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    "SELECT * FROM reminder_schedules WHERE schedule_id = ?",
                    (normalized_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise ScheduleNotFoundError(f"不存在提醒：{normalized_id}")
                current = _schedule_from_row(row)
                allowed = {
                    ScheduleStatus.ACTIVE: {ScheduleStatus.PAUSED, ScheduleStatus.CANCELLED},
                    ScheduleStatus.PAUSED: {ScheduleStatus.ACTIVE, ScheduleStatus.CANCELLED},
                    ScheduleStatus.COMPLETED: set(),
                    ScheduleStatus.CANCELLED: set(),
                }
                if status is current.status:
                    await connection.rollback()
                    return current
                if status not in allowed[current.status]:
                    raise ValueError(
                        f"提醒状态不能从{current.status.value}变为{status.value}。"
                    )
                await connection.execute(
                    "UPDATE reminder_schedules SET status = ?, updated_at = ? WHERE schedule_id = ?",
                    (status.value, now_text, normalized_id),
                )
                await connection.commit()
            except BaseException:
                if connection.in_transaction:
                    await connection.rollback()
                raise
        result = await self.get(normalized_id)
        assert result is not None
        return result

    async def materialize_due(self, *, now: datetime | None = None) -> int:
        """Create one delivery per due schedule and atomically advance it."""

        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        current_text = datetime_to_text(current)
        connection = self._connection_or_raise()
        created = 0
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """SELECT * FROM reminder_schedules
                       WHERE status = 'ACTIVE' AND next_run_at <= ?
                       ORDER BY next_run_at, schedule_id""",
                    (current_text,),
                ) as cursor:
                    rows = await cursor.fetchall()
                for row in rows:
                    schedule = _schedule_from_row(row)
                    scheduled_text = datetime_to_text(schedule.next_run_at)
                    run_id = f"srun_{uuid4().hex}"
                    cursor = await connection.execute(
                        """INSERT OR IGNORE INTO reminder_runs (
                            run_id,schedule_id,scheduled_for,status,created_at,updated_at,error
                        ) VALUES (?,?,?,'PENDING',?,?,NULL)""",
                        (run_id, schedule.schedule_id, scheduled_text, current_text, current_text),
                    )
                    created += max(0, cursor.rowcount)
                    following = next_occurrence(
                        schedule.next_run_at,
                        schedule.recurrence,
                        schedule.timezone_name,
                        after=current,
                    )
                    if following is None:
                        await connection.execute(
                            """UPDATE reminder_schedules
                               SET status='COMPLETED', updated_at=? WHERE schedule_id=?""",
                            (current_text, schedule.schedule_id),
                        )
                    else:
                        await connection.execute(
                            """UPDATE reminder_schedules
                               SET next_run_at=?, updated_at=? WHERE schedule_id=?""",
                            (datetime_to_text(following), current_text, schedule.schedule_id),
                        )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return created

    async def recover_claimed(self) -> int:
        connection = self._connection_or_raise()
        now_text = datetime_to_text(datetime.now(timezone.utc))
        async with self._lock:
            cursor = await connection.execute(
                "UPDATE reminder_runs SET status='PENDING', updated_at=? WHERE status='CLAIMED'",
                (now_text,),
            )
            await connection.commit()
            return max(0, cursor.rowcount)

    async def claim_next(self) -> tuple[ReminderSchedule, ReminderRun] | None:
        connection = self._connection_or_raise()
        now_text = datetime_to_text(datetime.now(timezone.utc))
        async with self._lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                async with connection.execute(
                    """SELECT * FROM reminder_runs WHERE status='PENDING'
                       ORDER BY created_at, run_id LIMIT 1"""
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    await connection.rollback()
                    return None
                await connection.execute(
                    "UPDATE reminder_runs SET status='CLAIMED', updated_at=? WHERE run_id=?",
                    (now_text, row["run_id"]),
                )
                async with connection.execute(
                    "SELECT * FROM reminder_schedules WHERE schedule_id=?",
                    (row["schedule_id"],),
                ) as cursor:
                    schedule_row = await cursor.fetchone()
                await connection.commit()
            except BaseException:
                if connection.in_transaction:
                    await connection.rollback()
                raise
        if schedule_row is None:
            raise ScheduleNotFoundError(str(row["schedule_id"]))
        run_record = dict(row)
        run_record["status"] = ReminderRunStatus.CLAIMED.value
        run_record["updated_at"] = now_text
        return _schedule_from_row(schedule_row), ReminderRun.from_dict(run_record)

    async def finish_run(
        self,
        run_id: str,
        status: ReminderRunStatus,
        *,
        error: str | None = None,
    ) -> None:
        if status not in {ReminderRunStatus.SUBMITTED, ReminderRunStatus.FAILED}:
            raise ValueError("提醒运行只能结束为SUBMITTED或FAILED。")
        connection = self._connection_or_raise()
        async with self._lock:
            await connection.execute(
                """UPDATE reminder_runs SET status=?, updated_at=?, error=?
                   WHERE run_id=? AND status='CLAIMED'""",
                (
                    status.value,
                    datetime_to_text(datetime.now(timezone.utc)),
                    error,
                    str(run_id).strip(),
                ),
            )
            await connection.commit()

    async def list_runs(self, schedule_id: str, *, limit: int = 20) -> list[ReminderRun]:
        if not 1 <= limit <= 100:
            raise ValueError("limit必须在1到100之间。")
        connection = self._connection_or_raise()
        async with self._lock:
            async with connection.execute(
                """SELECT * FROM reminder_runs WHERE schedule_id=?
                   ORDER BY created_at DESC, run_id DESC LIMIT ?""",
                (str(schedule_id).strip(), limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [_run_from_row(row) for row in rows]
