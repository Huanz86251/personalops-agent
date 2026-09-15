from types import SimpleNamespace
import pytest
from reporting.general_gate import general_review_reasons
from workers.general_completion import GeneralResult
from api_handoff import ApiHandoff

@pytest.mark.parametrize("change,expected", [
    ({}, []),
    ({"general_result": {"status": "FAILED"}}, ["not_completed"]),
    ({"general_result": {"status": "COMPLETED", "unresolved_items": ["missing"]}}, ["not_completed"]),
    ({"general_result": {"status": "COMPLETED", "forced_finalization": True}}, ["interrupted_or_budget_exhausted"]),
    ({"finish_reason": "BUDGET_EXHAUSTED"}, ["interrupted_or_budget_exhausted"]),
    ({"worker_finalize_requested": True, "worker_finalize_reason": "SCHEMA_REPAIR"}, []),
    ({"general_result": {"status": "COMPLETED", "files": [{"path": "/artifacts/a"}]}}, ["artifact_review"]),
])
def test_gate(change, expected):
    trace = {"general_result": {"status": "COMPLETED"}, **change}
    assert general_review_reasons(SimpleNamespace(artifact_outputs=[]), trace, SimpleNamespace(attempts=[])) == expected

@pytest.mark.parametrize("expected,actual", [(True,False),(False,True)])
def test_files_always_review(expected, actual):
    assert general_review_reasons(SimpleNamespace(artifact_outputs=[1] if expected else []),
        {"general_result": {"status": "COMPLETED"}},
        SimpleNamespace(attempts=[SimpleNamespace(resolved_artifacts=[1] if actual else [])])) == ["artifact_review"]

def test_schema_hides_harness_flag_and_has_no_fixed_version():
    assert "forced_finalization" not in GeneralResult.model_json_schema()["properties"]
    schema = ApiHandoff.model_json_schema()
    assert "version" not in schema["properties"]
    assert "next_action" not in schema.get("required", [])
