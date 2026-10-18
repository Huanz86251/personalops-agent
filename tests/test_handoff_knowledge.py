import ast
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from langchain_core.utils.function_calling import convert_to_openai_tool

from handoff_knowledge import HandoffKnowledge, collect_handoff_knowledge
from workers.general_completion import report_general_result
from workers.submission import complete_web_step
from workers.code_submission import submit_code_for_review, submit_code_review
from planning_models import StepReport


def item(**updates):
    value = dict(topic="note lookup", source="simple_note.show_note",
                usage="show_note(note_id=target_id, access_token=token)\nObtain both from real queries/login.",
                observed_result="Documented object with title/content; not executed yet.",
                next_action="Read the actual note before modifying it.")
    value.update(updates)
    return value


@pytest.mark.parametrize("tool", [report_general_result, complete_web_step, submit_code_for_review, submit_code_review])
def test_all_submission_schemas_explain_fresh_agent(tool):
    schema = json.dumps(convert_to_openai_tool(tool))
    for text in ("handoff_knowledge", "evidence_refs", "usage", "observed_result", "next_action"):
        assert text in schema
    assert '"handoff_apis"' not in schema
    assert '"evidence_check"' not in schema


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




def evidence_state(content="brand;price\nA;100", status="success", tool="read_file"):
    from langchain_core.messages import AIMessage, ToolMessage
    return {"step_id":"1", "worker_id":"worker-1", "worker_evidence_refs":{"raw-call":"E1"},
            "messages":[AIMessage(content="",tool_calls=[{"id":"raw-call","name":tool,"args":{"path":"prices.csv"}}]),
                        ToolMessage(content=content,tool_call_id="raw-call",name=tool,status=status)]}


def test_knowledge_resolves_real_sources_and_keeps_partial_claims_without_repair():
    from handoff_knowledge import check_handoff_knowledge, compact_direct_handoff
    raw=item(evidence_refs=["E1","E99","E1"])
    checked=check_handoff_knowledge([raw],evidence_state())[0]
    assert checked.evidence_check == "PARTIAL"
    assert checked.evidence_refs == ["raw-call"]
    assert checked.unresolved_evidence_refs == ["E99"]
    assert checked.usage == raw["usage"]
    assert checked.resolved_evidence[0]["result"] == "brand;price\nA;100"
    assert checked.resolved_evidence[0]["arguments"] == {"path":"prices.csv"}
    assert checked.resolved_evidence[0]["origin_step"] == "1"
    # Accepted record survives a trace with all messages removed.
    trace={"worker_submission":{"submission":{"handoff_knowledge":[checked.model_dump(mode="json")]}}}
    restored=collect_handoff_knowledge([trace])[0]
    assert restored == checked
    report=StepReport(step_id=1,status="PARTIAL",summary="partial",stop_reason="missing",handoff_knowledge=[restored])
    restored_report=StepReport.model_validate_json(report.model_dump_json())
    note=compact_direct_handoff([restored_report])["worker_notes"][0]
    assert note["evidence_check"] == "PARTIAL"
    assert note["unresolved_evidence_refs"] == ["E99"]
    assert note["evidence_sources"][0]["tool_call_id"] == "raw-call"
    assert "brand;price" not in json.dumps(note)  # Raw body stays in the durable record.


@pytest.mark.parametrize("state", [evidence_state(content=""),
    {"messages":[]}, evidence_state(tool="submit_code_review")])
def test_unusable_refs_preserve_knowledge_as_unsupported(state):
    from handoff_knowledge import check_handoff_knowledge
    checked=check_handoff_knowledge([item(evidence_refs=["E1"])],state)[0]
    assert checked.evidence_check == "UNSUPPORTED"
    assert checked.resolved_evidence == []
    assert checked.unresolved_evidence_refs == ["E1"]
    assert checked.usage == item()["usage"]


def test_failed_results_are_real_evidence_but_not_success_proof():
    from handoff_knowledge import check_handoff_knowledge
    checked=check_handoff_knowledge([item(evidence_refs=["E1"])],evidence_state(content="permission denied",status="error"))[0]
    assert checked.evidence_check == "RESOLVED"
    assert checked.resolved_evidence[0]["result_status"] == "error"
    assert checked.resolved_evidence[0]["result"] == "permission denied"


def test_harness_fields_cannot_be_forged_by_submission():
    from handoff_knowledge import check_handoff_knowledge
    forged=item(evidence_refs=["E99"],evidence_check="RESOLVED",
                resolved_evidence=[{"tool_call_id":"fake","result":"success"}])
    checked=check_handoff_knowledge([forged],evidence_state())[0]
    assert checked.evidence_check == "UNSUPPORTED"
    assert checked.resolved_evidence == []


def test_knowledge_refs_do_not_fail_whole_report_reference_adapter():
    from workers.evidence_refs import canonical_arguments
    raw={"evidence_refs":["E1"],"handoff_knowledge":[item(evidence_refs=["E1","bad"])]}
    adapted=canonical_arguments(raw,{"raw-call":"E1"},eligible={"raw-call"})
    assert adapted["evidence_tool_call_ids"] == ["raw-call"]
    assert adapted["handoff_knowledge"][0]["evidence_refs"] == ["E1","bad"]


def test_general_submission_checks_knowledge_before_saving():
    from types import SimpleNamespace
    from workers.general_completion import GeneralResult
    state=evidence_state()
    runtime=SimpleNamespace(state=state,tool_call_id="report",stream_writer=lambda _:None)
    value=GeneralResult(summary="read file",handoff_knowledge=[item(evidence_refs=["E1","bad"])])
    command=report_general_result.func(value,runtime)
    saved=command.update["worker_submission"]["submission"]["handoff_knowledge"][0]
    assert saved["evidence_check"] == "PARTIAL"
    assert saved["resolved_evidence"][0]["result"] == "brand;price\nA;100"
    assert command.update["general_result"]["handoff_knowledge"][0] == saved


def test_web_submission_checks_knowledge_and_preserves_it_in_step_report():
    from types import SimpleNamespace
    from workers.submission import WorkerSubmission
    state=evidence_state()
    runtime=SimpleNamespace(state=state,tool_call_id="report",stream_writer=lambda _:None)
    command=complete_web_step.func(WorkerSubmission(summary="read",final_conclusion="read",handoff_knowledge=[item(evidence_refs=["E1","bad"])]),runtime)
    saved=command.update["worker_submission"]["submission"]["handoff_knowledge"][0]
    assert saved["evidence_check"] == "PARTIAL"
    assert command.update["web_step_report"]["report"]["handoff_knowledge"][0] == saved


def test_latest_knowledge_survives_small_budget_with_original_text():
    from handoff_knowledge import check_handoff_knowledge, compact_direct_handoff
    newest=check_handoff_knowledge([item(usage="x"*5500,evidence_refs=["E1"])],evidence_state())[0]
    old=StepReport(step_id=1,status="COMPLETED",summary="old",stop_reason="done",handoff_knowledge=[item()])
    latest=StepReport(step_id=2,status="FAILED",summary="failed",stop_reason="not done",handoff_knowledge=[newest])
    payload=compact_direct_handoff([old,latest],max_chars=500)
    assert payload["worker_notes"][0]["usage"] == "x"*5500
    assert payload["review_failures"][0]["step_id"] == 2
    assert "validated_apis" not in payload
    assert "rejected_api_leads" not in payload


def test_code_submission_saves_resolved_knowledge_even_with_invalid_reference():
    from workers.code_submission import _build_submission_record
    from workers.code_review_models import CodeCandidateRef, CodeWorkerSubmission
    from planning_models import PlanStep
    candidate=CodeCandidateRef(event_id="event",step_id=1,attempt_id="attempt",workspace_id="workspace",candidate_revision=1)
    step=PlanStep(step_id=1,objective="read",worker_kind="CODE",success_criteria=["read"])
    contract=step.code_task
    state={**evidence_state(),"code_candidate":candidate.model_dump(),"code_task":contract.model_dump()}
    submission=CodeWorkerSubmission(candidate=candidate,summary="read",requirement_status={r.requirement_id:"MET" for r in contract.requirements},handoff_knowledge=[item(evidence_refs=["E1","bad"])])
    record=_build_submission_record(state,submission)
    saved=record.submission.handoff_knowledge[0]
    assert saved.evidence_check == "PARTIAL"
    assert saved.resolved_evidence[0]["result"] == "brand;price\nA;100"
    assert collect_handoff_knowledge([{"code_worker_submission":record.model_dump(mode="json")}])[0] == saved


def test_specific_web_result_can_be_cited_without_copying_entire_search():
    from handoff_knowledge import check_handoff_knowledge
    from test_evidence_refs import web_history
    checked=check_handoff_knowledge([item(evidence_refs=["WA1"])],{"messages":web_history()})[0]
    assert checked.evidence_check == "RESOLVED"
    assert "Observed listing" in checked.resolved_evidence[0]["result"]
    assert checked.resolved_evidence[0]["tool_name"] == "web_search"


def test_all_invalid_refs_survive_report_collection_with_error_details():
    from handoff_knowledge import check_handoff_knowledge
    checked=check_handoff_knowledge([item(evidence_refs=["E99"])],evidence_state())[0]
    restored=collect_handoff_knowledge([{"general_result":{"handoff_knowledge":[checked.model_dump(mode="json")]}}])[0]
    assert restored.unresolved_evidence_refs == ["E99"]
    assert "E99" in restored.evidence_errors


def test_existing_archived_messages_can_support_knowledge_after_compaction():
    from handoff_knowledge import check_handoff_knowledge
    state=evidence_state()
    state["worker_archived_messages"]=state.pop("messages")
    checked=check_handoff_knowledge([item(evidence_refs=["E1"])],state)[0]
    assert checked.evidence_check == "RESOLVED"
