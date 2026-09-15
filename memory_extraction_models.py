"""Typed schemas for the progressive long-term-memory extraction protocol."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from schema_utils import compact_schema


FrameType = Literal[
    "profile",
    "preference",
    "person_relation",
    "project",
    "task",
]
Level = Literal["low", "medium", "high"]
ImportanceLevel = Literal["low", "medium", "high", "urgent"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemoryFrame(StrictModel):
    """A cheap first-pass routing decision; it deliberately contains no quote."""

    frame_id: str = Field(min_length=1, max_length=40)
    frame_type: FrameType


class CandidateFramePlan(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=80)
    frames: list[MemoryFrame] = Field(default_factory=list, max_length=5)


class MemoryFramePlan(StrictModel):
    candidates: list[CandidateFramePlan] = Field(default_factory=list, max_length=32)


class TypedRecordBase(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=80)
    frame_id: str = Field(min_length=1, max_length=40)
    summary: str = Field(
        min_length=1,
        max_length=240,
        description="A concise standalone fact in Chinese; do not quote the source message.",
    )
    importance: ImportanceLevel = Field(
        description=(
            "Long-term usefulness: low, medium, high, or urgent. "
            "Use urgent only when delay can cause an imminent material consequence."
        )
    )
    confidence: Level = Field(
        description="How directly the source supports the record: low, medium, or high."
    )


class ProfileRecord(TypedRecordBase):
    record_type: Literal["profile"]
    field: Literal[
        "name",
        "preferred_name",
        "location",
        "job",
        "company",
        "language",
        "timezone",
        "education",
        "other",
    ] = Field(
        description=(
            "Use name only for an explicitly stated identity name; use "
            "preferred_name for how the user asks the assistant to address them."
        )
    )
    value: str = Field(min_length=1, max_length=160)
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None


class PreferenceRecord(TypedRecordBase):
    record_type: Literal["preference"]
    preference: Literal["like", "dislike", "prefer", "avoid"]
    topic: str = Field(min_length=1, max_length=160)
    scope: Literal["global", "work", "coding", "personal", "project", "other"]
    scope_name: str | None = Field(default=None, max_length=160)
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None


class PersonRelationRecord(TypedRecordBase):
    record_type: Literal["person_relation"]
    person_name: str = Field(min_length=1, max_length=120)
    relation: Literal[
        "mentor",
        "friend",
        "coworker",
        "leader",
        "team_member",
        "family",
        "partner",
        "client",
        "service_provider",
        "acquaintance",
        "other",
    ]
    other_relation: str | None = Field(default=None, max_length=80)
    state: Literal["current", "past", "uncertain"]
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None

    @model_validator(mode="after")
    def require_other_relation(self) -> "PersonRelationRecord":
        if self.relation == "other" and not (self.other_relation or "").strip():
            raise ValueError("other_relation is required when relation is other")
        if self.relation != "other" and self.other_relation is not None:
            raise ValueError("other_relation is only allowed when relation is other")
        return self


class ProjectRecord(TypedRecordBase):
    record_type: Literal["project"]
    project_name: str = Field(min_length=1, max_length=160)
    fact: Literal["role", "uses", "status", "goal", "rule", "deadline", "other"]
    value: str = Field(min_length=1, max_length=240)
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None


class TaskRecord(TypedRecordBase):
    record_type: Literal["task"]
    event: Literal["request", "order", "promise", "reminder", "update", "cancel"]
    from_person: str | None = Field(default=None, max_length=120)
    to_person: str = Field(min_length=1, max_length=120)
    action: str = Field(min_length=1, max_length=120)
    object: str | None = Field(default=None, max_length=240)
    project_name: str | None = Field(default=None, max_length=160)
    status: Literal["planned", "pending", "doing", "done", "cancelled", "unknown"]
    due_at: AwareDatetime | None = None


RECORD_MODELS: dict[str, type[TypedRecordBase]] = {
    "profile": ProfileRecord,
    "preference": PreferenceRecord,
    "person_relation": PersonRelationRecord,
    "project": ProjectRecord,
    "task": TaskRecord,
}


class TypedExtractionResult(StrictModel):
    """Light envelope; individual records are checked against selected schemas."""

    records: list[dict[str, Any]] = Field(default_factory=list, max_length=160)


def selected_output_schema(frame_types: set[str]) -> dict[str, Any]:
    """Build a compact oneOf schema containing only frame types selected in pass one."""

    selected = [RECORD_MODELS[name] for name in RECORD_MODELS if name in frame_types]
    record_schema: dict[str, Any]
    if selected:
        record_schema = {
            "oneOf": [compact_schema(model.model_json_schema()) for model in selected]
        }
    else:
        record_schema = {"type": "object"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["records"],
        "properties": {
            "records": {
                "type": "array",
                "items": record_schema,
                "maxItems": 160,
            }
        },
    }


def validate_typed_records(
    payload: dict[str, Any],
    *,
    allowed_frame_types: set[str],
) -> list[TypedRecordBase]:
    envelope = TypedExtractionResult.model_validate(payload)
    records: list[TypedRecordBase] = []
    for raw_record in envelope.records:
        record_type = raw_record.get("record_type")
        if record_type not in allowed_frame_types:
            raise ValueError(f"record_type is not selected by pass one: {record_type!r}")
        model = RECORD_MODELS.get(str(record_type))
        if model is None:
            raise ValueError(f"unsupported record_type: {record_type!r}")
        records.append(model.model_validate(raw_record))
    return records
