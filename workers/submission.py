"""Structured Worker submission and Tool Call evidence resolution."""

from __future__ import annotations

import json
import base64
import hashlib
import mimetypes
from pathlib import Path
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic.json_schema import SkipJsonSchema
from api_handoff import ApiHandoffList, api_field
from handoff_knowledge import HandoffKnowledge, knowledge_field
from pydantic import BaseModel, Field, field_validator
from prompt_loader import load_prompt
from artifact_models import (
    DownloadedArtifactRecord,
    ResolvedArtifactCandidate,
    WorkerArtifactCandidate,
)
from workers.plan_challenge import PlanChallenge
from deepagents.backends.utils import file_data_to_string
from run_workspace import (
    RUN_WORKSPACE_ROOT,
    initialize_run_workspace,
    materialize_worker_artifact,
)


SUBMIT_FOR_REVIEW_NAME = "submit_for_review"
CONTROL_TOOL_NAMES = frozenset(
    {
        "publish_worker_progress",
        SUBMIT_FOR_REVIEW_NAME,
        "report_general_result",
    }
)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def _normalize_unique_strings(values: list[str]) -> list[str]:
    normalized_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_values.append(normalized)
    return normalized_values


class WorkerCriterionClaim(BaseModel):
    """One Worker claim mapped to the exact Step success criterion."""

    criterion_id: str | None = None

    criterion: str = Field(min_length=1, max_length=1200)
    evidence_tool_call_ids: list[str] = Field(
        default_factory=list,
        description="先引用支持本项判断的真实Tool Call结果，再填写conclusion。",
    )
    conclusion: str = Field(min_length=1, max_length=2000)

    @field_validator("criterion", "conclusion")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError("claim text cannot be empty")
        return normalized

    @field_validator("evidence_tool_call_ids")
    @classmethod
    def normalize_evidence_ids(cls, values: list[str]) -> list[str]:
        return _normalize_unique_strings(values)


class WorkerSubmission(BaseModel):
    """The Worker''s final claim; independent review remains authoritative."""

    handoff_apis: ApiHandoffList = api_field()
    handoff_knowledge: SkipJsonSchema[list[HandoffKnowledge]] = knowledge_field()

    plan_challenge: PlanChallenge | None = Field(
        default=None,
        description=(
            "仅当当前Step或范围关系与用户原话或真实只读事实发生实质冲突时填写；"
            "提交后停止本次执行，由独立Reviewer判断是否请求Scheduler重规划。"
        ),
    )

    summary: str = Field(min_length=1, max_length=2000)
    criterion_claims: list[WorkerCriterionClaim] = Field(
        description="先逐项覆盖冻结标准并引用必要的真实Tool Call结果，再填写final_conclusion。",
        default_factory=list,
        max_length=12,
    )
    artifact_candidates: list[WorkerArtifactCandidate] = Field(
        default_factory=list,
        max_length=12,
    )
    unresolved_items: list[str] = Field(default_factory=list, max_length=12)
    final_conclusion: str = Field(min_length=1, max_length=1600)

    @field_validator("summary", "final_conclusion")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError("submission text cannot be empty")
        return normalized

    @field_validator("unresolved_items")
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        return _normalize_unique_strings(values)

class ResolvedToolEvidence(BaseModel):
    """A real Tool request/result pair resolved from the current Attempt."""

    tool_call_id: str
    tool_name: str
    arguments: Any = None
    result: str
    result_chars: int = Field(ge=0)


class WorkerSubmissionRecord(BaseModel):
    """Harness-authored envelope persisted in Worker graph state."""

    submitted_at: datetime
    worker_id: str | None = None
    event_id: str | None = None
    step_id: str | None = None
    total_tool_calls: int = Field(ge=0)
    submission: WorkerSubmission
    resolved_evidence: list[ResolvedToolEvidence] = Field(default_factory=list)
    resolved_artifacts: list[ResolvedArtifactCandidate] = Field(
        default_factory=list,
        max_length=12,
    )


def _state_file_bytes(file_data: Any) -> bytes:
    content = file_data_to_string(file_data)
    if str(file_data.get("encoding", "utf-8")) == "base64":
        return base64.standard_b64decode(content)
    return content.encode("utf-8")


def resolve_artifact_candidates(
    state: Any,
    candidates: Sequence[WorkerArtifactCandidate],
) -> list[ResolvedArtifactCandidate]:
    """Resolve model proposals against checkpoint or runtime-authored facts."""

    state_files = state.get("files", {}) or {}
    raw_downloads = state.get("worker_downloaded_artifacts", []) or []
    downloads: dict[str, DownloadedArtifactRecord] = {}
    for raw_record in raw_downloads:
        record = DownloadedArtifactRecord.model_validate(raw_record)
        downloads[record.candidate_id] = record

    resolved: list[ResolvedArtifactCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            raise ValueError(
                f"duplicate artifact candidate ID: {candidate.candidate_id}"
            )
        seen.add(candidate.candidate_id)

        common = {
            "candidate_id": candidate.candidate_id,
            "output_id": candidate.output_id,
            "kind": candidate.kind,
            "description": candidate.description,
            "evidence_tool_call_ids": candidate.evidence_tool_call_ids,
            "relevant_paths": candidate.relevant_paths,
        }
        if candidate.kind == "WORKSPACE_FILE":
            path = str(candidate.path)
            file_data = state_files.get(path)
            if file_data is None:
                raise ValueError(f"workspace artifact does not exist: {path}")
            payload = _state_file_bytes(file_data)
            run_id = str(
                state.get("event_id")
                or state.get("planning_run_id")
                or ""
            ).strip()
            worker_id = str(state.get("worker_id") or "").strip()
            if not run_id or not worker_id:
                raise ValueError(
                    "workspace artifact requires run and Worker identity"
                )
            storage_root = Path(
                state.get("run_storage_root") or RUN_WORKSPACE_ROOT
            )
            layout = initialize_run_workspace(
                run_id,
                storage_root=storage_root,
            )
            frozen = materialize_worker_artifact(
                layout=layout,
                worker_id=worker_id,
                candidate_id=candidate.candidate_id,
                filename=Path(path).name,
                payload=payload,
            )
            resolved.append(
                ResolvedArtifactCandidate(
                    **common,
                    location=path,
                    storage_path=str(frozen),
                    size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    media_type=mimetypes.guess_type(path)[0],
                )
            )
        elif candidate.kind == "DOWNLOADED_FILE":
            record = downloads.get(candidate.candidate_id)
            if record is None:
                raise ValueError(
                    "unknown downloaded artifact candidate ID: "
                    f"{candidate.candidate_id}"
                )
            if record.tool_call_id not in candidate.evidence_tool_call_ids:
                raise ValueError(
                    "downloaded artifact must cite its download Tool Call ID"
                )
            stored = Path(record.storage_path)
            if not stored.is_file():
                raise ValueError(
                    f"downloaded artifact is no longer available: {candidate.candidate_id}"
                )
            if record.run_id is not None:
                from task_files import read_download
                payload = read_download("/downloads/" + record.candidate_id, state)
            else:
                from file_limits import TASK_FILE_MAX_BYTES
                if stored.stat().st_size > TASK_FILE_MAX_BYTES:
                    raise ValueError("Downloaded artifact exceeds 20 MiB")
                payload = stored.read_bytes()
            if len(payload) != record.size_bytes or hashlib.sha256(payload).hexdigest() != record.sha256:
                raise ValueError(
                    f"downloaded artifact changed after download: {candidate.candidate_id}"
                )
            resolved.append(
                ResolvedArtifactCandidate(
                    **common,
                    location=record.storage_path,
                    storage_path=record.storage_path,
                    size_bytes=record.size_bytes,
                    sha256=record.sha256,
                    media_type=record.media_type,
                    source_url=record.source_url,
                )
            )
        else:
            location = str(candidate.repository_url)
            if candidate.commit:
                location = f"{location}@{candidate.commit}"
            elif candidate.repository_ref:
                location = f"{location}@{candidate.repository_ref}"
            resolved.append(
                ResolvedArtifactCandidate(
                    **common,
                    location=location,
                    source_url=candidate.repository_url,
                    repository_ref=candidate.repository_ref,
                    commit=candidate.commit,
                )
            )
    return resolved


def _tool_call_value(tool_call: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(tool_call, dict):
        if field_name in tool_call:
            return tool_call[field_name]
        function_data = tool_call.get("function")
        if isinstance(function_data, dict):
            return function_data.get(field_name, default)
        return default
    return getattr(tool_call, field_name, default)


def _tool_message_value(message: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(field_name, default)
    return getattr(message, field_name, default)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def resolve_tool_evidence(
    messages: Sequence[Any],
    requested_ids: Sequence[str],
) -> list[ResolvedToolEvidence]:
    """Resolve cited Tool Calls without interpreting their business content."""

    normalized_ids = _normalize_unique_strings(list(requested_ids))
    calls: dict[str, tuple[str, Any]] = {}
    results: dict[str, str] = {}

    for message in messages:
        raw_tool_calls = _tool_message_value(message, "tool_calls", []) or []
        for raw_call in raw_tool_calls:
            call_id = _normalize_text(
                _tool_call_value(raw_call, "id", "")
                or _tool_call_value(raw_call, "tool_call_id", "")
            )
            if not call_id:
                continue
            tool_name = _normalize_text(
                _tool_call_value(raw_call, "name", "")
            )
            arguments = _tool_call_value(raw_call, "args")
            if arguments is None:
                arguments = _tool_call_value(raw_call, "arguments")
            previous = calls.get(call_id)
            current = (tool_name, arguments)
            if previous is not None and previous != current:
                raise ValueError(f"duplicate Tool Call ID has conflicting data: {call_id}")
            calls[call_id] = current

        result_call_id = _normalize_text(
            _tool_message_value(message, "tool_call_id", "")
        )
        if result_call_id:
            result_text = _content_to_text(
                _tool_message_value(message, "content", "")
            )
            previous_result = results.get(result_call_id)
            if previous_result is not None and previous_result != result_text:
                raise ValueError(
                    "duplicate Tool Result ID has conflicting data: "
                    f"{result_call_id}"
                )
            results[result_call_id] = result_text

    resolved: list[ResolvedToolEvidence] = []
    for call_id in normalized_ids:
        call = calls.get(call_id)
        if call is None:
            raise ValueError(f"unknown Tool Call ID: {call_id}")
        tool_name, arguments = call
        if tool_name in CONTROL_TOOL_NAMES:
            raise ValueError(
                f"control tool cannot be cited as evidence: {call_id}"
            )
        if call_id not in results:
            raise ValueError(f"Tool Call has no result: {call_id}")
        result = results[call_id]
        if not result.strip():
            raise ValueError(f"Tool Call result is empty: {call_id}")
        resolved.append(
            ResolvedToolEvidence(
                tool_call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                result_chars=len(result),
            )
        )
    return resolved


@tool(
    SUBMIT_FOR_REVIEW_NAME,
    description=load_prompt(
        "workers/submit_for_review_tool"
    ),
)
def submit_for_review(
    submission: WorkerSubmission,
    runtime: ToolRuntime,
) -> Command:
    """Submit final claims and cited Tool Calls for independent Step review."""

    state = runtime.state
    if submission.plan_challenge is not None and int(
        state.get("worker_plan_challenges_remaining", 0) or 0
    ) <= 0:
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    "Plan challenge rejected by Harness: the global challenge budget is exhausted. "
                    "Continue within the current Step when possible; otherwise submit the actual "
                    "unresolved result with plan_challenge=null. No Reviewer or Scheduler was called."
                ),
                tool_call_id=runtime.tool_call_id or SUBMIT_FOR_REVIEW_NAME,
                name=SUBMIT_FOR_REVIEW_NAME,
                status="error",
            )]
        })
    requested_ids = [
        evidence_id
        for claim in submission.criterion_claims
        for evidence_id in claim.evidence_tool_call_ids
    ]
    requested_ids.extend(
        evidence_id
        for candidate in submission.artifact_candidates
        for evidence_id in candidate.evidence_tool_call_ids
    )
    if submission.plan_challenge is not None:
        requested_ids.extend(submission.plan_challenge.evidence_tool_call_ids)
    from reporting.criteria import criterion_claim_repair_feedback, validate_worker_claim_coverage
    try:
        validate_worker_claim_coverage(state, submission.criterion_claims)
        resolved_evidence = resolve_tool_evidence(
            list(state.get("worker_archived_messages", [])) + list(state.get("messages", [])),
            requested_ids,
        )
        resolved_artifacts = resolve_artifact_candidates(
            state,
            submission.artifact_candidates,
        )
    except ValueError as error:
        return Command(
            update={
                "worker_finalize_requested": True,
                "worker_finalize_reason": "SCHEMA_REPAIR",
                "messages": [
                    ToolMessage(
                        content=(
                            criterion_claim_repair_feedback(state, error)
                        ),
                        tool_call_id=(
                            runtime.tool_call_id or SUBMIT_FOR_REVIEW_NAME
                        ),
                        name=SUBMIT_FOR_REVIEW_NAME,
                        status="error",
                    )
                ]
            }
        )

    record = WorkerSubmissionRecord(
        submitted_at=datetime.now(timezone.utc),
        worker_id=state.get("worker_id"),
        event_id=state.get("event_id"),
        step_id=state.get("step_id"),
        total_tool_calls=max(
            int(state.get("worker_total_tool_calls", 0) or 0),
            0,
        ),
        submission=submission,
        resolved_evidence=resolved_evidence,
        resolved_artifacts=resolved_artifacts,
    )
    record_value = record.model_dump(mode="json")
    runtime.stream_writer(
        {
            "type": "worker_submission",
            "record": record_value,
        }
    )
    return Command(
        update={
            "worker_review_requested": True,
            "worker_submission": record_value,
            "messages": [
                ToolMessage(
                    content=(
                        "Worker submission accepted for independent review. "
                        f"Resolved {len(resolved_evidence)} cited Tool Calls "
                        f"and {len(resolved_artifacts)} artifact candidates."
                    ),
                    tool_call_id=(
                        runtime.tool_call_id or SUBMIT_FOR_REVIEW_NAME
                    ),
                )
            ],
        }
    )


__all__ = [
    "SUBMIT_FOR_REVIEW_NAME",
    "ResolvedToolEvidence",
    "WorkerCriterionClaim",
    "WorkerSubmission",
    "WorkerSubmissionRecord",
    "resolve_artifact_candidates",
    "resolve_tool_evidence",
    "submit_for_review",
]
