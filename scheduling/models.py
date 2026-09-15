"""Provider-free domain objects for local scheduled reminders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class Recurrence(str, Enum):
    ONCE = "ONCE"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"


class DeliveryTarget(str, Enum):
    WINDOWS = "WINDOWS"
    FEISHU = "FEISHU"
    AGENT_EVENT = "AGENT_EVENT"


class ScheduleStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class ReminderRunStatus(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    SUBMITTED = "SUBMITTED"
    FAILED = "FAILED"


def as_utc(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")
    return value.astimezone(timezone.utc)


def datetime_to_text(value: datetime) -> str:
    return as_utc(value, field_name="datetime").isoformat().replace("+00:00", "Z")


def datetime_from_text(value: object, *, field_name: str) -> datetime:
    normalized = str(value).strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime.") from error
    return as_utc(parsed, field_name=field_name)


def parse_run_at(value: str) -> datetime:
    """Parse a model-supplied ISO timestamp; offsets are mandatory."""

    normalized = str(value).strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            "run_at必须是带时区偏移的ISO-8601时间，例如"
            "2026-09-07T09:00:00+08:00。"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            "run_at必须包含时区偏移，例如+08:00；不能提交无时区时间。"
        )
    return parsed.astimezone(timezone.utc)


def validate_timezone(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("timezone_name不能为空。")
    try:
        ZoneInfo(normalized)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"无法识别IANA时区：{normalized}") from error
    return normalized


def next_occurrence(
    scheduled_for: datetime,
    recurrence: Recurrence,
    timezone_name: str,
    *,
    after: datetime,
) -> datetime | None:
    """Advance recurring wall-clock time and skip an offline backlog."""

    if recurrence is Recurrence.ONCE:
        return None
    zone = ZoneInfo(validate_timezone(timezone_name))
    local = as_utc(scheduled_for, field_name="scheduled_for").astimezone(zone)
    local_after = as_utc(after, field_name="after").astimezone(zone)
    step = timedelta(days=1 if recurrence is Recurrence.DAILY else 7)
    candidate = local
    while candidate <= local_after:
        candidate += step
    return candidate.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class ReminderSchedule:
    schedule_id: str
    title: str
    message: str
    delivery_target: DeliveryTarget
    reply_target_id: str | None
    conversation_id: str | None
    recurrence: Recurrence
    timezone_name: str
    next_run_at: datetime
    status: ScheduleStatus
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        for field_name in ("schedule_id", "title", "message"):
            normalized = str(getattr(self, field_name)).strip()
            if not normalized:
                raise ValueError(f"{field_name}不能为空。")
            object.__setattr__(self, field_name, normalized)
        reply_target_id = (
            None if self.reply_target_id is None
            else str(self.reply_target_id).strip() or None
        )
        conversation_id = (
            None if self.conversation_id is None
            else str(self.conversation_id).strip() or None
        )
        if self.delivery_target is DeliveryTarget.FEISHU and reply_target_id is None:
            raise ValueError("飞书提醒必须绑定可信回复会话。")
        if self.delivery_target is DeliveryTarget.AGENT_EVENT:
            if reply_target_id is None or conversation_id is None:
                raise ValueError("定时Agent任务必须绑定飞书回复会话和Conversation。")
        object.__setattr__(self, "reply_target_id", reply_target_id)
        object.__setattr__(self, "conversation_id", conversation_id)
        object.__setattr__(self, "timezone_name", validate_timezone(self.timezone_name))
        object.__setattr__(self, "next_run_at", as_utc(self.next_run_at, field_name="next_run_at"))
        object.__setattr__(self, "created_at", as_utc(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", as_utc(self.updated_at, field_name="updated_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "title": self.title,
            "message": self.message,
            "delivery_target": self.delivery_target.value,
            "reply_target_id": self.reply_target_id,
            "conversation_id": self.conversation_id,
            "recurrence": self.recurrence.value,
            "timezone_name": self.timezone_name,
            "next_run_at": datetime_to_text(self.next_run_at),
            "status": self.status.value,
            "created_at": datetime_to_text(self.created_at),
            "updated_at": datetime_to_text(self.updated_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReminderSchedule":
        return cls(
            schedule_id=str(value["schedule_id"]),
            title=str(value["title"]),
            message=str(value["message"]),
            delivery_target=DeliveryTarget(
                str(value.get("delivery_target") or DeliveryTarget.WINDOWS.value)
            ),
            reply_target_id=value.get("reply_target_id"),
            conversation_id=value.get("conversation_id"),
            recurrence=Recurrence(str(value["recurrence"])),
            timezone_name=str(value["timezone_name"]),
            next_run_at=datetime_from_text(value["next_run_at"], field_name="next_run_at"),
            status=ScheduleStatus(str(value["status"])),
            created_at=datetime_from_text(value["created_at"], field_name="created_at"),
            updated_at=datetime_from_text(value["updated_at"], field_name="updated_at"),
        )


@dataclass(frozen=True, slots=True)
class ReminderRun:
    run_id: str
    schedule_id: str
    scheduled_for: datetime
    status: ReminderRunStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schedule_id": self.schedule_id,
            "scheduled_for": datetime_to_text(self.scheduled_for),
            "status": self.status.value,
            "created_at": datetime_to_text(self.created_at),
            "updated_at": datetime_to_text(self.updated_at),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReminderRun":
        return cls(
            run_id=str(value["run_id"]),
            schedule_id=str(value["schedule_id"]),
            scheduled_for=datetime_from_text(value["scheduled_for"], field_name="scheduled_for"),
            status=ReminderRunStatus(str(value["status"])),
            created_at=datetime_from_text(value["created_at"], field_name="created_at"),
            updated_at=datetime_from_text(value["updated_at"], field_name="updated_at"),
            error=None if value.get("error") is None else str(value["error"]),
        )


def create_schedule(
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
    created_at = as_utc(now or datetime.now(timezone.utc), field_name="now")
    return ReminderSchedule(
        schedule_id=f"sch_{uuid4().hex}",
        title=title,
        message=message,
        delivery_target=delivery_target,
        reply_target_id=reply_target_id,
        conversation_id=conversation_id,
        recurrence=recurrence,
        timezone_name=timezone_name,
        next_run_at=run_at,
        status=ScheduleStatus.ACTIVE,
        created_at=created_at,
        updated_at=created_at,
    )
