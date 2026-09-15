"""Application-owned trace hierarchy. Observations never alter model messages."""
from contextvars import ContextVar
from functools import wraps
import inspect
from pathlib import Path
from collections.abc import Mapping

from observability import trace_span, set_span_output, set_span_attributes

TRACE_IDS = ContextVar("personalops_trace_ids", default={})
ROLE_NAMES = {
    "skill_selector": "Skill Selector",
    "scope_resolver": "Scope Resolver",
    "scheduler": "Scheduler", "code_scheduler": "Scheduler / Code Recovery",
    "replanner": "Scheduler / Replan", "final_reviewer": "Scheduler / Final Review",
    "worker_leader": "Scheduler / Worker Leader", "general": "General Agent",
    "web": "Web Agent", "code": "Code Worker", "reviewer": "Code Reviewer", "code_reviewer": "Code Reviewer",
    "reporter": "General Reviewer", "web_reporter": "Web Reporter",
    "extraction": "Memory Extractor", "title": "Conversation Title",
    "summary": "Conversation Summary", "general_summary": "General Summary",
    "web_summary": "Web Summary", "code_summary": "Code Worker Summary",
    "reviewer_summary": "Code Reviewer Summary", "code_reviewer_summary": "Code Reviewer Summary",
}

ID_KEYS = ("event_id", "step_id", "worker_id", "attempt_id", "workspace_id",
           "candidate_revision", "thread_id", "planning_run_id", "pair_id",
           "code_runtime_session_id", "repair_round", "scheduler_epoch", "attempt_no", "current_step_attempt")

def identities(value):
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        value = asdict(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        return {}
    result = {k: value[k] for k in ID_KEYS if value.get(k) is not None}
    for key in ("code_candidate", "candidate", "code_review_loop", "configurable", "current_step"):
        result.update(identities(value.get(key)))
    return result


def summary(value):
    """Compact outputs; full tool/model evidence lives on its own leaf span."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if hasattr(value, "update") and not isinstance(value, Mapping):
        value = value.update
    if isinstance(value, Mapping):
        keys = (*ID_KEYS, "status", "final_status", "final_answer", "overall_stop_reason", "planning_failure",
                "code_review_loop", "code_review_report", "code_worker_submission", "code_scheduler_decision_applied",
                "code_publication_receipt", "code_integration_status", "code_attempt_final_record",
                "summary", "reason", "verdict", "action", "exit_code", "truncated", "phase",
                "output", "selected", "role", "mode", "selection_method", "snapshot_sha256", "catalog_sha256", "model_calls",
                "objective", "success_criteria", "requirements", "validation_expectations", "current_step", "remaining_steps",
                "current_step_attempt", "completed_step_reports", "code_task", "worker_kind", "findings", "check_results",
                "verification_summary", "failed_test_summaries", "requirement_status", "changed_files", "approved_artifact_paths",
                "published_artifact_paths", "delivery_location", "publication_id", "applied_revision", "recommended_action",
                "worker_instruction", "reviewer_instruction", "repair_rounds", "statement", "description", "evidence_refs",
                "candidate_id", "frame_id", "record_type", "candidates", "frames", "confidence", "importance", "field", "value",
                "raw_user_text", "queued_at", "base_revision", "applied_revision", "manifest_id", "artifacts", "sha256")
        return {k: value[k] for k in keys if k in value} or {"fields": sorted(map(str, value))}
    if isinstance(value, (list, tuple)):
        return {"count": len(value), "items": [summary(v) for v in value[:10]]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return summary(asdict(value))
    return {"type": type(value).__name__}


def operation(name, *, fields=(), kind="chain"):
    """Trace a concrete runtime boundary with inherited async-safe identities."""
    def decorate(fn):
        signature = inspect.signature(fn)
        def prepare(args, kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            ids = dict(TRACE_IDS.get())
            for key in ("state", "input_state", "candidate", "pair", "config"):
                ids.update(identities(bound.get(key)))
            ids.update({k: bound[k] for k in ID_KEYS if k in bound})
            backend = bound.get("self")
            if getattr(backend, "_id", None):
                ids["sandbox_id"] = str(backend._id)
            selected = {k: summary(bound[k]) for k in fields if k in bound}
            return ids, selected
        def finish(span, result):
            output = summary(result)
            set_span_output(span, output)
            if isinstance(output, dict):
                status = output.get("final_status") or output.get("status") or output.get("verdict")
                if status:
                    set_span_attributes(span, **{"business.status": str(status)})
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_run(*args, **kwargs):
                ids, value = prepare(args, kwargs)
                token = TRACE_IDS.set(ids)
                try:
                    with trace_span(name, kind=kind, input_value=value, attributes={"runtime." + k: v for k, v in ids.items()}) as span:
                        result = await fn(*args, **kwargs)
                        finish(span, result)
                        return result
                finally:
                    TRACE_IDS.reset(token)
            return async_run
        @wraps(fn)
        def run(*args, **kwargs):
            ids, value = prepare(args, kwargs)
            token = TRACE_IDS.set(ids)
            try:
                with trace_span(name, kind=kind, input_value=value, attributes={"runtime." + k: v for k, v in ids.items()}) as span:
                    result = fn(*args, **kwargs)
                    finish(span, result)
                    return result
            finally:
                TRACE_IDS.reset(token)
        return run
    return decorate
