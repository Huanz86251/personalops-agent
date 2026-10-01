"""Deterministic normalization for model-authored structured reports.

The model owns semantic judgments.  The Harness owns harmless container-shape
repair and aggregate statuses that can be derived from criterion-level facts.
"""

from __future__ import annotations

import json
import logging
import types
from typing import Annotated, Any, Iterable, Literal, Union, get_args, get_origin

from pydantic import Field, TypeAdapter, ValidationError


logger = logging.getLogger("agent")


EvidenceRefText = Annotated[str, Field(min_length=1, max_length=160)]
ConciseReportItem = Annotated[str, Field(min_length=1, max_length=1200)]
ReportItem = Annotated[str, Field(min_length=1, max_length=2000)]


def coerce_sequence(value: Any) -> Any:
    """Wrap one model-emitted item in a list without inventing content."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    return [value]


def coerce_text_sequence(value: Any) -> Any:
    """Normalize scalar/list text containers; mappings remain validation errors."""

    values = coerce_sequence(value)
    if not isinstance(values, list):
        return values
    normalized: list[Any] = []
    for item in values:
        if isinstance(item, dict):
            normalized.append(item)
        elif item is None:
            continue
        else:
            normalized.append(str(item))
    return normalized


def coerce_json_mapping(value: Any) -> Any:
    """Decode a JSON-object string while preserving invalid values for validation.

    Some compatible structured-output providers serialize a nested object one
    extra time. Decoding that harmless container error is deterministic; plain
    prose, arrays and malformed JSON remain validation errors rather than being
    guessed into a semantic object.
    """

    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        decoded = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return value
    return decoded if isinstance(decoded, dict) else value


def _sequence_origin(annotation: Any) -> type | None:
    origin = get_origin(annotation)
    if origin in {list, tuple, set, frozenset}:
        return origin
    if origin is Annotated:
        args = get_args(annotation)
        return _sequence_origin(args[0]) if args else None
    if origin in {Union, types.UnionType}:
        kinds = {
            kind
            for argument in get_args(annotation)
            if argument is not type(None)
            and (kind := _sequence_origin(argument)) is not None
        }
        return next(iter(kinds)) if len(kinds) == 1 else None
    return None


def coerce_declared_sequences(model_cls: Any, value: Any) -> Any:
    """Wrap scalar values for declared list/tuple fields before validation.

    This repairs a common structured-output shape error without changing any
    semantic content. Mappings are deliberately not unpacked or guessed.
    """

    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    for name, field in getattr(model_cls, "model_fields", {}).items():
        if name not in normalized or normalized[name] is None:
            continue
        kind = _sequence_origin(field.annotation)
        if kind is None:
            continue
        item = normalized[name]
        if isinstance(item, (list, tuple, set, frozenset)):
            continue
        normalized[name] = [item]
    return normalized


def drop_invalid_auxiliary_fields(
    model_cls: Any,
    value: Any,
    field_names: Iterable[str],
) -> Any:
    """Drop only explicitly allow-listed optional, non-control fields.

    A malformed explanation or supplementary list must not force another
    model call, while routing, status, delivery, and selection fields must
    never disappear silently. Callers therefore provide a small per-schema
    allow-list. Required fields and all other fields remain strict.
    """

    if not isinstance(value, dict):
        return value
    normalized = dict(coerce_declared_sequences(model_cls, value))
    dropped: list[str] = []
    fields = getattr(model_cls, "model_fields", {})
    for name in field_names:
        if name not in normalized:
            continue
        field = fields.get(name)
        if field is None:
            raise RuntimeError(
                f"Auxiliary field policy references unknown {model_cls.__name__}.{name}"
            )
        if field.is_required():
            raise RuntimeError(
                f"Auxiliary field policy cannot drop required {model_cls.__name__}.{name}"
            )
        annotation: Any = field.annotation
        if field.metadata:
            annotation = Annotated[annotation, *field.metadata]
        adapter = TypeAdapter(annotation)
        try:
            adapter.validate_python(normalized[name])
        except (TypeError, ValueError, ValidationError):
            # Existing report contracts intentionally accept scalar numbers
            # in text sequences by stringifying them. Preserve that harmless
            # repair before deciding the whole auxiliary field is malformed.
            text_normalized = coerce_text_sequence(normalized[name])
            try:
                adapter.validate_python(text_normalized)
            except (TypeError, ValueError, ValidationError):
                normalized.pop(name, None)
                dropped.append(name)
            else:
                normalized[name] = text_normalized
    if dropped:
        logger.info(
            "Harness省略无效可选Schema字段 | schema=%s | fields=%s",
            getattr(model_cls, "__name__", str(model_cls)),
            dropped,
        )
    return normalized


def normalize_supervisor_decision_payload(value: Any) -> Any:
    """Repair only SupervisorDecision contradictions fixed by its own schema.

    This is intentionally not a generic ``extra='ignore'`` escape hatch.  It
    handles two provider/model mistakes for which the contract already gives a
    unique answer:

    * every target set must emit ``target_entity``; a missing per-set
      ``result_entity`` can therefore be copied from that canonical field, and
      an identical legacy top-level copy is redundant;
    * an explicitly SINGLE step executes its own objective, so zero or one
      ``worker_assignments`` entry cannot describe a parallel branch and is
      discarded.  Two or more entries remain an error because the intended
      execution mode is then genuinely ambiguous.

    Conflicting entity names, absent canonical fields, and other semantic
    errors are preserved for normal validation/model repair.
    """

    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    raw_steps = normalized.get("steps")
    if isinstance(raw_steps, dict):
        steps: list[Any] = [raw_steps]
        preserve_container = False
    elif isinstance(raw_steps, list):
        steps = list(raw_steps)
        preserve_container = True
    else:
        return normalized

    repaired_steps: list[Any] = []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            repaired_steps.append(raw_step)
            continue
        step = dict(raw_step)

        mode = str(step.get("execution_mode") or "").strip().upper()
        assignments = step.get("worker_assignments")
        assignment_count = (
            len(assignments)
            if isinstance(assignments, list)
            else 1 if isinstance(assignments, dict) else 0
        )
        if mode == "SINGLE" and assignment_count <= 1:
            step.pop("worker_assignments", None)

        raw_selection = step.get("target_selection")
        if isinstance(raw_selection, dict):
            selection = dict(raw_selection)
            target_entity = str(selection.get("target_entity") or "").strip()
            legacy_entity = str(selection.get("result_entity") or "").strip()
            raw_sets = selection.get("sets")
            set_items = (
                [raw_sets]
                if isinstance(raw_sets, dict)
                else list(raw_sets) if isinstance(raw_sets, list) else None
            )
            if target_entity and set_items:
                repaired_sets: list[Any] = []
                for raw_set in set_items:
                    if not isinstance(raw_set, dict):
                        repaired_sets.append(raw_set)
                        continue
                    target_set = dict(raw_set)
                    if not str(target_set.get("result_entity") or "").strip():
                        target_set["result_entity"] = target_entity
                    repaired_sets.append(target_set)
                selection["sets"] = (
                    repaired_sets[0]
                    if isinstance(raw_sets, dict)
                    else repaired_sets
                )
                set_entities = {
                    str(item.get("result_entity") or "").strip()
                    for item in repaired_sets
                    if isinstance(item, dict)
                }
                if legacy_entity == target_entity and set_entities == {target_entity}:
                    selection.pop("result_entity", None)
            step["target_selection"] = selection

        repaired_steps.append(step)

    normalized["steps"] = repaired_steps if preserve_container else repaired_steps[0]
    return normalized


CriterionStatusValue = Literal["MET", "PARTIAL", "NOT_MET", "UNKNOWN"]
StepStatusValue = Literal["COMPLETED", "PARTIAL", "BLOCKED", "FAILED"]
FinalStatusValue = Literal["COMPLETED", "PARTIAL", "FAILED"]


def derive_step_status(statuses: Iterable[str]) -> StepStatusValue:
    """Derive one Step status from criterion-level judgments."""

    values = [str(item).strip().upper() for item in statuses]
    if not values:
        return "BLOCKED"
    if all(item == "MET" for item in values):
        return "COMPLETED"
    if any(item in {"MET", "PARTIAL"} for item in values):
        return "PARTIAL"
    if any(item == "UNKNOWN" for item in values):
        return "BLOCKED"
    return "FAILED"


def derive_final_status(statuses: Iterable[str]) -> FinalStatusValue:
    """Final answers cannot be BLOCKED: uncertainty is disclosed as PARTIAL."""

    values = [str(item).strip().upper() for item in statuses]
    if values and all(item == "MET" for item in values):
        return "COMPLETED"
    if values and all(item == "NOT_MET" for item in values):
        return "FAILED"
    return "PARTIAL"


__all__ = [
    "ConciseReportItem",
    "EvidenceRefText",
    "FinalStatusValue",
    "ReportItem",
    "StepStatusValue",
    "coerce_sequence",
    "coerce_text_sequence",
    "coerce_declared_sequences",
    "coerce_json_mapping",
    "drop_invalid_auxiliary_fields",
    "derive_final_status",
    "derive_step_status",
    "normalize_supervisor_decision_payload",
]
