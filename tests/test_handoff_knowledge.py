import ast
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from langchain_core.utils.function_calling import convert_to_openai_tool

from handoff_knowledge import HandoffKnowledge, collect_handoff_knowledge
from workers.general_completion import report_general_result
from workers.submission import submit_for_review
from workers.code_submission import submit_code_for_review, submit_code_review
from planning_models import StepReport


def item(**updates):
    return dict(topic="note lookup", source="simple_note.show_note",
                usage="show_note(note_id=target_id, access_token=token)\nObtain both from real queries/login.",
                observed_result="Documented object with title/content; not executed yet.",
                next_action="Read the actual note before modifying it.", **updates)


@pytest.mark.parametrize("tool", [report_general_result, submit_for_review, submit_code_for_review, submit_code_review])
def test_all_submission_schemas_explain_fresh_agent(tool):
    schema = json.dumps(convert_to_openai_tool(tool))
    for text in ("handoff_apis", "documentation", "parameters", "call_example", "next_action"):
        assert text in schema
    assert '"handoff_knowledge"' not in schema


def test_accepted_knowledge_survives_report_serialization_and_corrects_prior_item():
    first = item()
    corrected = {**first, "observed_result": "Executed successfully; title/content returned."}
    traces = [{"worker_submission": {"submission": {"handoff_knowledge": [first]}}},
              {"code_worker_submission": {"submission": {"handoff_knowledge": [first]}},
               "code_review_report": {"handoff_knowledge": [corrected]}}]
    knowledge = collect_handoff_knowledge(traces)
    assert len(knowledge) == 1
    report = StepReport(step_id=1, status="COMPLETED", summary="done", stop_reason="done", handoff_knowledge=knowledge)
    restored = StepReport.model_validate_json(report.model_dump_json())
    assert restored.handoff_knowledge[0].model_dump(exclude_defaults=True) == first
    assert traces[0]["worker_submission"]["submission"]["handoff_knowledge"][0] == first


def test_name_only_is_not_a_valid_handoff():
    with pytest.raises(ValidationError):
        HandoffKnowledge(topic="show_note")


def test_old_report_without_knowledge_remains_loadable():
    assert StepReport(step_id=1, status="COMPLETED", summary="old", stop_reason="done").handoff_knowledge == []


def test_next_step_receives_usage_and_return_structure():
    from planning_graph import _build_step_instruction
    from planning_models import PlanStep, PlanningContextPack
    from test_planning_handoff_publication import planning_settings
    knowledge = HandoffKnowledge(**item())
    report = StepReport(step_id=1, status="COMPLETED", summary="done", stop_reason="done", handoff_knowledge=[knowledge])
    state = {"context": PlanningContextPack(current_time="now", user_request="Continue task"),
             "plan_objective": "Continue task", "completed_step_reports": [report]}
    prompt = _build_step_instruction(state, PlanStep(step_id=2, worker_kind="GENERAL",
        objective="Continue", success_criteria=["done"]), 1,
        model_limit=4, tool_limit=4, planning=planning_settings())
    assert "show_note(note_id=target_id, access_token=token)" in prompt
    assert "Documented object with title/content; not executed yet." in prompt


def test_scheduler_api_suggestion_is_explicitly_non_authoritative_to_worker():
    from planning_graph import _build_step_instruction
    from planning_models import PlanStep, PlanningContextPack
    from test_planning_handoff_publication import planning_settings

    state = {
        "context": PlanningContextPack(current_time="now", user_request="处理记录"),
        "plan_objective": "处理记录",
        "completed_step_reports": [],
    }
    step = PlanStep.model_validate({
        "step_id": 1,
        "objective": "处理记录",
        "success_criteria": ["处理完成"],
        "worker_kind": "GENERAL",
        "api_suggestion": {
            "use_reason": "已见目录表明它能读取候选记录。",
            "api_name": "service.list_records",
        },
    })
    prompt = _build_step_instruction(
        state, step, 1, model_limit=4, tool_limit=4, planning=planning_settings()
    )
    assert "Scheduler API建议：低置信、必须自行查证" in prompt
    assert "service.list_records" in prompt
    assert "如果不准确，请忽略该建议" in prompt


def test_replanned_skill_context_contains_only_compact_decision_facts():
    from planning_graph import _replanned_skill_selection_context
    from planning_models import PlanStep, PlanningContextPack

    step = PlanStep(
        step_id=3,
        objective="重新读取并导出",
        success_criteria=["导出通过验收"],
        worker_kind="GENERAL",
    )
    state = {
        "context": PlanningContextPack(current_time="now", user_request="导出后关闭账户"),
        "completed_step_reports": [
            StepReport(step_id=1, status="COMPLETED", summary="已读取账户", stop_reason="done")
        ],
        "replan_history": [{
            "request_reason": "旧写法未通过格式验收",
            "remaining_steps": [step.model_dump(mode="json")],
        }],
    }
    context = _replanned_skill_selection_context(state, step)
    assert context == {
        "user_request": "导出后关闭账户",
        "accepted_steps": [{"step_id": 1, "status": "COMPLETED", "summary": "已读取账户"}],
        "previous_failure": "旧写法未通过格式验收",
        "new_step_id": 3,
        "objective": "重新读取并导出",
        "success_criteria": ["导出通过验收"],
    }


def test_skill_examples_parse_and_wrong_call_is_not_executable():
    text = Path("skills/appworld/appworld-execute-api/SKILL.md").read_text(encoding="utf-8")
    calls = []
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        calls.extend(ast.unparse(n.func) for n in ast.walk(ast.parse(block)) if isinstance(n, ast.Call))
    assert "apis.simple_note.read_note" not in calls
    assert "apis.simple_note.show_note" in calls


def test_next_step_receives_one_compact_direct_api_handoff():
    from planning_graph import _build_step_instruction
    from planning_models import PlanStep, PlanningContextPack
    from test_planning_handoff_publication import planning_settings
    api = {
        "name": "service.update",
        "purpose": "update record",
        "documentation": {"tool_call_id": "doc", "pointer": "/name"},
        "parameters": [
            {
                "name": "record_id",
                "required": True,
                "purpose": "target",
                "source": {"tool_call_id": "query", "pointer": "/id"},
            }
        ],
        "call_example": "service.update(record_id=record_id)",
        "next_action": "apply then verify",
    }
    report = StepReport(
        step_id=1,
        status="COMPLETED",
        summary="done",
        stop_reason="done",
        handoff_apis=[api],
    )
    state = {
        "context": PlanningContextPack(current_time="now", user_request="Continue task"),
        "plan_objective": "Continue task",
        "completed_step_reports": [report],
    }
    prompt = _build_step_instruction(
        state,
        PlanStep(step_id=2, worker_kind="GENERAL", objective="Continue", success_criteria=["done"]),
        1,
        model_limit=4,
        tool_limit=4,
        planning=planning_settings(),
    )
    assert "前序Worker直接交接" in prompt
    assert '"validated_apis"' in prompt
    assert "service.update(record_id=record_id)" in prompt
    prior_reports = prompt.split("此前StepReport：", 1)[1]
    assert "handoff_apis" not in prior_reports
    assert "handoff_knowledge" not in prior_reports


def test_latest_direct_handoff_ignores_legacy_budget_and_keeps_whole_entries():
    from handoff_knowledge import compact_direct_handoff
    api = {
        "name": "service.update",
        "purpose": "x" * 500,
        "documentation": {"tool_call_id": "doc", "pointer": ""},
        "parameters": [],
        "call_example": "service.update()",
    }
    report = StepReport(
        step_id=1,
        status="COMPLETED",
        summary="done",
        stop_reason="done",
        handoff_apis=[api],
    )
    payload = compact_direct_handoff([report], max_chars=120)
    assert len(payload["validated_apis"] ) == 1
    assert payload["validated_apis"][0]["purpose"] == "x" * 500
    assert "omitted_entries" not in payload
    assert len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) > 120


def test_next_step_sees_rejected_api_lead_and_validation_reason():
    from api_handoff import ApiHandoffReceipt
    from handoff_knowledge import compact_direct_handoff
    api = {
        "name": "service.update",
        "purpose": "update record",
        "documentation": {"tool_call_id": "doc", "pointer": ""},
        "parameters": [],
        "call_example": "service.update()",
    }
    report = StepReport(
        step_id=1,
        status="PARTIAL",
        summary="handoff validation failed",
        stop_reason="needs correction",
        handoff_api_receipts=[
            ApiHandoffReceipt(
                api=api,
                validation_status="REJECTED",
                validation_error="documentation source call missing",
            )
        ],
    )
    payload = compact_direct_handoff([report])
    assert payload["validated_apis"] == []
    assert payload["rejected_api_leads"][0]["name"] == "service.update"
    assert "missing" in payload["rejected_api_leads"][0]["validation_error"]


def test_failed_review_is_directly_visible_without_api_handoff():
    from handoff_knowledge import compact_direct_handoff
    report = StepReport(
        step_id=3,
        status="FAILED",
        summary="verification did not pass",
        stop_reason="reviewer found missing read-back",
        unresolved_items=["write succeeded but final state was not verified"],
        errors=["verification call returned an error"],
    )
    payload = compact_direct_handoff([report])
    assert payload["review_failures"][0]["step_id"] == 3
    assert payload["review_failures"][0]["status"] == "FAILED"
    assert "read-back" in payload["review_failures"][0]["stop_reason"]



def _rejected_receipt(name, *, next_action=""):
    from api_handoff import ApiHandoffReceipt
    return ApiHandoffReceipt(
        api={
            "name": name,
            "purpose": f"use {name}",
            "documentation": {"tool_call_id": f"doc-{name}", "pointer": ""},
            "parameters": [],
            "call_example": f"{name}()",
            "next_action": next_action,
        },
        validation_status="REJECTED",
        validation_error="documentation source call missing",
    )


def test_direct_handoff_keeps_latest_failure_before_old_rejected_leads():
    from handoff_knowledge import compact_direct_handoff

    old = StepReport(
        step_id=1,
        status="PARTIAL",
        summary="old",
        stop_reason="old read failed",
        handoff_api_receipts=[
            _rejected_receipt(f"service.read_{index}")
            for index in range(5)
        ],
    )
    latest = StepReport(
        step_id=2,
        status="FAILED",
        summary="latest",
        stop_reason="write still needs to run",
        unresolved_items=["create the missing records"],
        handoff_api_receipts=[
            _rejected_receipt(
                "service.create_record",
                next_action="Create the missing records, then verify.",
            )
        ],
    )
    payload = compact_direct_handoff([old, latest], max_chars=900)

    assert payload["review_failures"][0]["step_id"] == 2
    assert any(
        item["name"] == "service.create_record"
        for item in payload["rejected_api_leads"]
    )
    assert payload["omitted_entries"] >= 1
    assert next(
        item for item in payload["rejected_api_leads"]
        if item["name"] == "service.create_record"
    )["next_action"] == "Create the missing records, then verify."
