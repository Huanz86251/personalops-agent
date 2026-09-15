from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from workers.general_completion import GeneralResult
from workers.submission import WorkerSubmission, WorkerSubmissionRecord, resolve_tool_evidence
from reporting.models import ReviewAttempt


def evidence(count):
    calls = [{"id": f"call-{i}", "name": "query", "args": {}} for i in range(count)]
    messages = [AIMessage(content="", tool_calls=calls)]
    messages += [ToolMessage(content="observed result", tool_call_id=c["id"]) for c in calls]
    return resolve_tool_evidence(messages, [c["id"] for c in calls])


@pytest.mark.parametrize("count", [7, 12])
def test_more_than_six_survives_submission_and_review_packet(count):
    refs = [f"call-{i}" for i in range(count)]
    GeneralResult(status="COMPLETED", summary="done", evidence_tool_call_ids=refs)
    claims = [{"criterion": str(i), "conclusion": "observed", "evidence_tool_call_ids": [ref]} for i, ref in enumerate(refs)]
    submission = WorkerSubmission(summary="done", final_conclusion="done", criterion_claims=claims)
    resolved = evidence(count)
    record = WorkerSubmissionRecord(submitted_at=datetime.now(timezone.utc), total_tool_calls=count,
                                    submission=submission, resolved_evidence=resolved)
    attempt = ReviewAttempt(attempt=1, submission_source="WORKER", finish_reason="done",
        stop_reason="done", summary="done", final_conclusion="done", resolved_evidence=record.resolved_evidence)
    assert len(ReviewAttempt.model_validate_json(attempt.model_dump_json()).resolved_evidence) == count


def test_evidence_count_is_unbounded_at_each_boundary():
    count = 30
    refs = [f"call-{i}" for i in range(count)]
    general = GeneralResult(status="COMPLETED", summary="done", evidence_tool_call_ids=refs)
    claims = [{"criterion": "criterion", "conclusion": "observed", "evidence_tool_call_ids": refs}]
    submission = WorkerSubmission(summary="done", final_conclusion="done", criterion_claims=claims)
    resolved = evidence(count)
    record = WorkerSubmissionRecord(
        submitted_at=datetime.now(timezone.utc), total_tool_calls=count,
        submission=submission, resolved_evidence=resolved,
    )
    attempt = ReviewAttempt(
        attempt=1, submission_source="WORKER", finish_reason="done", stop_reason="done",
        summary="done", final_conclusion="done", resolved_evidence=resolved,
    )
    assert len(general.evidence_tool_call_ids) == count
    assert len(submission.criterion_claims[0].evidence_tool_call_ids) == count
    assert len(record.resolved_evidence) == count
    assert len(attempt.resolved_evidence) == count


def test_unknown_reference_still_rejected():
    with pytest.raises(ValueError, match="unknown Tool Call ID"):
        resolve_tool_evidence([], ["fake"])
