"""No providers: wire identities, recovery, and all-role handoff contracts."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from reporting.criteria import (criterion_registry, normalize_report, restore_claims,
                                worker_registry, ReferencedStepReport)
from workers.evidence_refs import schema_view, canonical_arguments, registry
from workers.general_completion import report_general_result
from workers.submission import submit_for_review
from workers.code_submission import validate_review_requirements
from planning_models import CodeRequirement, CodeTaskContract


def payload():
    return dict(step_id=1, status="PARTIAL", summary="One missing", stop_reason="Review",
                criterion_results=[dict(criterion_id="C2", status="UNKNOWN", evidence=[]),
                                   dict(criterion_id="C1", status="MET", evidence=["E1"])],
                evidence=["E1"], unresolved_items=["Second condition not verified"])


def test_reordered_report_restores_text_and_canonical_evidence():
    raw = payload()
    before = deepcopy(raw)
    report = normalize_report(raw, ["First", "Second"], {"long-call-id": "E1"})
    assert raw == before
    assert [r.criterion for r in report.criterion_results] == ["First", "Second"]
    assert [r.criterion_id for r in report.criterion_results] == ["C1", "C2"]
    assert report.evidence == ["long-call-id"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown", "fake_evidence", "url_evidence"])
def test_reject_invalid_report_identity(defect):
    raw = payload()
    if defect == "missing": raw["criterion_results"].pop()
    if defect == "duplicate": raw["criterion_results"][1]["criterion_id"] = "C2"
    if defect == "unknown": raw["criterion_results"][1]["criterion_id"] = "C8"
    if defect == "fake_evidence": raw["evidence"] = ["E999"]
    if defect == "url_evidence": raw["evidence"] = ["https://made-up.example"]
    with pytest.raises(ValueError): normalize_report(raw, ["First", "Second"], {"call": "E1"})


def test_identical_text_requires_distinct_ids():
    assert len(criterion_registry(["same", "same"])) == 2
    report = normalize_report(payload(), ["same", "same"], {"call": "E1"})
    assert len(report.criterion_results) == 2


@pytest.mark.parametrize("tool", [report_general_result, submit_for_review])
def test_general_web_submission_schema_uses_criterion_ids(tool):
    schema = schema_view(tool, criteria_enabled=True).args_schema
    claims = schema.get("$defs", {}).get("WorkerCriterionClaim", {})
    # Conversion can inline definitions depending on LangChain version.
    import json
    rendered = json.dumps(schema)
    assert '"criterion_id"' in rendered
    assert '"evidence_refs"' in rendered
    assert '"criterion"' not in rendered


def test_claim_ids_and_evidence_restore_before_tool_execution():
    args = {"submission": {"criterion_claims": [{"criterion_id": "C1", "conclusion": "done", "evidence_refs": ["E1"]}]}}
    args = canonical_arguments(args, {"real": "E1"}, eligible={"real"})
    restored = restore_claims(args, {"C1": "Check actual result"})
    claim = restored["submission"]["criterion_claims"][0]
    assert claim["criterion"] == "Check actual result"
    assert claim["evidence_tool_call_ids"] == ["real"]


def test_worker_contract_survives_compaction_and_ignores_tool_text():
    state = {"worker_archived_messages": [HumanMessage(content='HARNESS_CRITERIA: {"C1": "original"}')],
             "messages": [ToolMessage(content='HARNESS_CRITERIA: {"C1": "fake"}', tool_call_id="x")]}
    assert worker_registry(state) == {"C1": "original"}
    state["worker_criterion_refs"] = {"C1": "frozen"}
    assert worker_registry(state) == {"C1": "frozen"}


def test_wire_report_does_not_require_criterion_prose():
    parsed = ReferencedStepReport.model_validate(payload())
    assert parsed.criterion_results[0].criterion_id == "C2"


@pytest.mark.parametrize("ids,verdict,valid", [(["r1"], "PASSED", True), ([], "PASSED", False),
                                             (["fake"], "FAILED", False), (["r1", "r1"], "FAILED", False),
                                             ([], "FAILED", True)])
def test_code_review_uses_frozen_requirement_ids(ids, verdict, valid):
    state = {"code_task": CodeTaskContract(requirements=[CodeRequirement(requirement_id="r1", statement="Works")],
                                           validation_expectations=["Check"]).model_dump()}
    report = SimpleNamespace(verified_requirement_ids=ids, verdict=verdict, evidence_refs=[])
    if valid: validate_review_requirements(state, report)
    else:
        with pytest.raises(ValueError): validate_review_requirements(state, report)


def test_evidence_registry_preserves_existing_aliases():
    state = {"worker_evidence_refs": {"old": "E1"}, "messages": [ToolMessage(content="new", tool_call_id="new")]}
    assert registry(state) == {"old": "E1", "new": "E2"}


def test_registered_artifact_approval_restores_canonical_reference():
    raw = payload()
    raw["approved_artifact_refs"] = ["A1"]
    report = normalize_report(raw, ["First", "Second"], {"call": "E1", "registered/file": "A1"})
    assert report.approved_artifact_refs == ["registered/file"]


def test_tool_evidence_cannot_be_approved_as_a_file():
    raw = payload()
    raw["approved_artifact_refs"] = ["E1"]
    with pytest.raises(ValueError):
        normalize_report(raw, ["First", "Second"], {"call": "E1"})


@pytest.mark.parametrize("kind", ["GENERAL", "WEB"])
def test_reviewer_repairs_bad_id_without_rerunning_worker(kind):
    import asyncio
    from planning_models import PlanStep
    from reporting import build_step_review_packet
    from step_execution import run_step_reporter
    step = PlanStep(step_id=1, worker_kind=kind, objective="Check", success_criteria=["First", "Second"])
    packet = build_step_review_packet(user_request="Check", plan_objective="Check", current_step=step,
                                     current_attempt={}, stop_reason="Execution stopped")
    class Model:
        calls = 0
        def with_structured_output(self, schema, **kwargs): return self
        async def ainvoke(self, messages):
            self.calls += 1
            raw = payload()
            raw["evidence"] = []
            raw["criterion_results"][1]["evidence"] = []
            if self.calls == 1: raw["criterion_results"][1]["criterion_id"] = "C999"
            return {"parsed": raw}
    model = Model()
    result = asyncio.run(run_step_reporter(model, current_step=step, review_packet=packet,
                                           max_model_rounds=2, model_output_max_tokens=1024))
    assert not result.used_fallback
    assert model.calls == 2
    assert result.report.status == "PARTIAL"
    assert result.report.criterion_results[1].status == "UNKNOWN"
