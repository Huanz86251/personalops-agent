"""Event and run state shared by channel adapters and the scheduler.

The objects in this module deliberately contain no Feishu, SQLite, model, or
LangGraph runtime dependency.  Their ``to_dict`` output contains JSON-native
values only, so the same representation can be stored in SQLite or placed in
LangGraph state without serializing callbacks, locks, or live runtime objects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


class EventAction(str, Enum):
    """The four explicit scheduling actions supported by the first version."""

    QUEUE = "QUEUE"
    INSERT = "INSERT"
    REPLACE = "REPLACE"
    CANCEL = "CANCEL"


class EventOrigin(str, Enum):
    """The adapter that accepted the event."""

    FEISHU = "FEISHU"
    DESKTOP = "DESKTOP"
    SYSTEM = "SYSTEM"


class EventStatus(str, Enum):
    """Whether the scheduler has applied an immutable input event."""

    PENDING = "PENDING"
    HANDLING = "HANDLING"
    APPLIED = "APPLIED"
    FAILED = "FAILED"


class RunStatus(str, Enum):
    """Current execution state of an event that owns a LangGraph run."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"


EVENT_STATUS_TRANSITIONS: dict[EventStatus, frozenset[EventStatus]] = {
    EventStatus.PENDING: frozenset(
        {
            EventStatus.HANDLING,
            EventStatus.FAILED,
        }
    ),
    EventStatus.HANDLING: frozenset(
        {
            # A process that died while handling an event may return it to the
            # pending queue during startup recovery.
            EventStatus.PENDING,
            EventStatus.APPLIED,
            EventStatus.FAILED,
        }
    ),
    EventStatus.APPLIED: frozenset(),
    EventStatus.FAILED: frozenset(),
}


RUN_STATUS_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.CANCELLED,
            RunStatus.SUPERSEDED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.PAUSED,
            RunStatus.COMPLETED,
            RunStatus.CANCELLED,
            RunStatus.SUPERSEDED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.PAUSED: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.CANCELLED,
            RunStatus.SUPERSEDED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
    RunStatus.SUPERSEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
}


def _normalize_identifier(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_optional_identifier(
    value: object | None,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return _normalize_identifier(value, field_name=field_name)


def _as_utc(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_to_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _datetime_from_text(value: object, *, field_name: str) -> datetime:
    normalized = _normalize_identifier(value, field_name=field_name)
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime.") from exc
    return _as_utc(parsed, field_name=field_name)


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A persisted user or system instruction.

    ``event_id`` is also the stable identity used by the matching execution
    record.  ``target_event_id`` points to the event whose run is affected by
    INSERT, REPLACE, or CANCEL; it is not merely the previously received event.
    """

    event_id: str
    conversation_id: str
    action: EventAction
    status: EventStatus
    payload_text: str
    target_event_id: str | None
    origin: EventOrigin
    received_at: datetime
    status_changed_at: datetime
    reply_target_id: str | None = None
    result_code: str | None = None

    def __post_init__(self) -> None:
        event_id = _normalize_identifier(self.event_id, field_name="event_id")
        conversation_id = _normalize_identifier(
            self.conversation_id,
            field_name="conversation_id",
        )
        target_event_id = _normalize_optional_identifier(
            self.target_event_id,
            field_name="target_event_id",
        )
        payload_text = str(self.payload_text).strip()
        result_code = _normalize_optional_identifier(
            self.result_code,
            field_name="result_code",
        )
        reply_target_id = _normalize_optional_identifier(
            self.reply_target_id,
            field_name="reply_target_id",
        )
        received_at = _as_utc(self.received_at, field_name="received_at")
        status_changed_at = _as_utc(
            self.status_changed_at,
            field_name="status_changed_at",
        )

        if self.action is EventAction.CANCEL:
            if payload_text:
                raise ValueError("CANCEL events cannot contain task payload text.")
            if target_event_id is None:
                raise ValueError("CANCEL events require target_event_id.")
        elif not payload_text:
            raise ValueError(f"{self.action.value} events require payload text.")

        if self.action is EventAction.REPLACE and target_event_id is None:
            raise ValueError("REPLACE events require target_event_id.")

        if target_event_id == event_id:
            raise ValueError("An event cannot target itself.")

        if status_changed_at < received_at:
            raise ValueError("status_changed_at cannot be before received_at.")

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "conversation_id", conversation_id)
        object.__setattr__(self, "payload_text", payload_text)
        object.__setattr__(self, "target_event_id", target_event_id)
        object.__setattr__(self, "received_at", received_at)
        object.__setattr__(self, "status_changed_at", status_changed_at)
        object.__setattr__(self, "reply_target_id", reply_target_id)
        object.__setattr__(self, "result_code", result_code)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-native representation suitable for LangGraph state."""

        return {
            "event_id": self.event_id,
            "conversation_id": self.conversation_id,
            "action": self.action.value,
            "status": self.status.value,
            "payload_text": self.payload_text,
            "target_event_id": self.target_event_id,
            "origin": self.origin.value,
            "received_at": _datetime_to_text(self.received_at),
            "status_changed_at": _datetime_to_text(self.status_changed_at),
            "reply_target_id": self.reply_target_id,
            "result_code": self.result_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentEvent":
        """Rebuild an event from SQLite JSON or checkpoint state."""

        return cls(
            event_id=str(value["event_id"]),
            conversation_id=str(value["conversation_id"]),
            action=EventAction(str(value["action"])),
            status=EventStatus(str(value["status"])),
            payload_text=str(value.get("payload_text", "")),
            target_event_id=value.get("target_event_id"),
            origin=EventOrigin(str(value["origin"])),
            received_at=_datetime_from_text(
                value["received_at"],
                field_name="received_at",
            ),
            status_changed_at=_datetime_from_text(
                value["status_changed_at"],
                field_name="status_changed_at",
            ),
            reply_target_id=value.get("reply_target_id"),
            result_code=value.get("result_code"),
        )


@dataclass(frozen=True, slots=True)
class EventRun:
    """Execution state for an event that owns a LangGraph run."""

    event_id: str
    status: RunStatus
    status_changed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_id",
            _normalize_identifier(self.event_id, field_name="event_id"),
        )
        object.__setattr__(
            self,
            "status_changed_at",
            _as_utc(self.status_changed_at, field_name="status_changed_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-native representation suitable for persistence."""

        return {
            "event_id": self.event_id,
            "status": self.status.value,
            "status_changed_at": _datetime_to_text(self.status_changed_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventRun":
        return cls(
            event_id=str(value["event_id"]),
            status=RunStatus(str(value["status"])),
            status_changed_at=_datetime_from_text(
                value["status_changed_at"],
                field_name="status_changed_at",
            ),
        )


def create_agent_event(
    *,
    conversation_id: str,
    action: EventAction,
    payload_text: str,
    origin: EventOrigin,
    target_event_id: str | None = None,
    reply_target_id: str | None = None,
    event_id: str | None = None,
    received_at: datetime | None = None,
) -> AgentEvent:
    """Create a new pending event with one authoritative receive time."""

    resolved_received_at = _as_utc(
        received_at or _utc_now(),
        field_name="received_at",
    )
    return AgentEvent(
        event_id=event_id or f"evt_{uuid4().hex}",
        conversation_id=conversation_id,
        action=action,
        status=EventStatus.PENDING,
        payload_text=payload_text,
        target_event_id=target_event_id,
        origin=origin,
        received_at=resolved_received_at,
        status_changed_at=resolved_received_at,
        reply_target_id=reply_target_id,
    )


def create_event_run(
    event: AgentEvent,
    *,
    created_at: datetime | None = None,
) -> EventRun:
    """Create the queued run owned by a task-producing event."""

    if event.action is EventAction.CANCEL:
        raise ValueError("CANCEL events do not own an Agent run.")
    resolved_created_at = _as_utc(
        created_at or event.received_at,
        field_name="created_at",
    )
    if resolved_created_at < event.received_at:
        raise ValueError("A run cannot be created before its event was received.")
    return EventRun(
        event_id=event.event_id,
        status=RunStatus.QUEUED,
        status_changed_at=resolved_created_at,
    )


def transition_event_status(
    event: AgentEvent,
    new_status: EventStatus,
    *,
    changed_at: datetime | None = None,
    result_code: str | None = None,
) -> AgentEvent:
    """Return a new event after validating an atomic status transition."""

    if new_status not in EVENT_STATUS_TRANSITIONS[event.status]:
        raise ValueError(
            f"Illegal event transition: {event.status.value} -> {new_status.value}."
        )
    resolved_changed_at = _as_utc(
        changed_at or _utc_now(),
        field_name="changed_at",
    )
    if resolved_changed_at < event.status_changed_at:
        raise ValueError("Event status time cannot move backwards.")
    return replace(
        event,
        status=new_status,
        status_changed_at=resolved_changed_at,
        result_code=result_code,
    )


def transition_run_status(
    run: EventRun,
    new_status: RunStatus,
    *,
    changed_at: datetime | None = None,
) -> EventRun:
    """Return a new run after validating a legal lifecycle transition."""

    if new_status not in RUN_STATUS_TRANSITIONS[run.status]:
        raise ValueError(
            f"Illegal run transition: {run.status.value} -> {new_status.value}."
        )
    resolved_changed_at = _as_utc(
        changed_at or _utc_now(),
        field_name="changed_at",
    )
    if resolved_changed_at < run.status_changed_at:
        raise ValueError("Run status time cannot move backwards.")
    return replace(
        run,
        status=new_status,
        status_changed_at=resolved_changed_at,
    )


def build_planning_thread_id_from_parts(
    *,
    conversation_id: str,
    event_id: str,
) -> str:
    """Build the stable LangGraph checkpoint cursor from persisted identities."""

    normalized_conversation_id = conversation_id.strip()
    normalized_event_id = event_id.strip()
    if not normalized_conversation_id:
        raise ValueError("conversation_id cannot be empty.")
    if not normalized_event_id:
        raise ValueError("event_id cannot be empty.")
    return f"planning:{normalized_conversation_id}:{normalized_event_id}"


def build_planning_thread_id(event: AgentEvent) -> str:
    """Build the stable LangGraph checkpoint cursor for one event run."""

    return build_planning_thread_id_from_parts(
        conversation_id=event.conversation_id,
        event_id=event.event_id,
    )
