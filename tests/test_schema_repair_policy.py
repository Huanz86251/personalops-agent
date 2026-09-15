from pydantic import BaseModel, Field, ValidationError

from reporting.criteria import validate_worker_claim_coverage
from schema_utils import (
    is_schema_repairable_error,
    minimal_schema_example,
    schema_repair_feedback,
)
from workers.submission import WorkerCriterionClaim


class NestedRow(BaseModel):
    criterion_id: str
    evidence_refs: list[str] = Field(default_factory=list)


class SampleReport(BaseModel):
    status: str
    rows: list[NestedRow]


def claim(criterion_id, evidence_count=0):
    return WorkerCriterionClaim(
        criterion_id=criterion_id,
        criterion=f"criterion {criterion_id}",
        conclusion="checked",
        evidence_tool_call_ids=[f"call-{index}" for index in range(evidence_count)],
    )


def test_schema_feedback_contains_required_shape_without_failed_values():
    schema = SampleReport.model_json_schema()
    feedback = schema_repair_feedback(
        schema_name="SampleReport",
        schema=schema,
        error_text="rows.0.criterion_id is required",
    )
    assert "status" in feedback
    assert "rows" in feedback
    assert "rows.0.criterion_id is required" in feedback
    assert minimal_schema_example(schema) == {"status": "<string>", "rows": []}


def test_retry_classifier_separates_schema_errors_from_timeouts():
    try:
        SampleReport.model_validate({"status": "done", "rows": [{"evidence_refs": []}]})
    except ValidationError as error:
        assert is_schema_repairable_error(error)
    else:
        raise AssertionError("fixture must fail validation")
    assert not is_schema_repairable_error(TimeoutError("request timed out"))
    assert not is_schema_repairable_error(ConnectionError("connection reset"))


def test_frozen_criteria_require_exact_coverage():
    state = {"worker_criterion_refs": {"C1": "first", "C2": "second"}}
    try:
        validate_worker_claim_coverage(state, [claim("C1")])
    except ValueError as error:
        assert "missing=['C2']" in str(error)
    else:
        raise AssertionError("missing claim must be rejected")
    validate_worker_claim_coverage(state, [claim("C1"), claim("C2")])


def test_evidence_references_have_no_arbitrary_count_limit():
    item = claim("C1", evidence_count=30)
    assert len(item.evidence_tool_call_ids) == 30
