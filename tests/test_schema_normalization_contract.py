import asyncio
from types import SimpleNamespace

from langchain.messages import AIMessage, ToolMessage

from planning_models import FinalReviewDecision, StepReport
from reporting.criteria import ReferencedStepReport
from reporting.materials import ReviewWithReads
from workers.code_review_models import (
    CodeCandidateRef,
    CodeReviewFinding,
    CodeWorkerSubmission,
)
from workers.control_protocol import defer_terminal_call_if_mixed
from workers.general_completion import GeneralResult
from workers.submission import WorkerSubmission


def _properties(model):
    return model.model_json_schema()["properties"]


def test_harness_owned_aggregate_statuses_are_hidden_from_model_schemas():
    assert "status" not in _properties(ReferencedStepReport)
    assert "status" not in _properties(GeneralResult)
    assert "status" not in _properties(WorkerSubmission)
    assert "status" not in _properties(FinalReviewDecision)


def test_final_status_is_derived_from_criterion_rows():
    decision = FinalReviewDecision.model_validate({
        "review_reason": "All visible criteria were checked.",
        "criterion_reviews": {
            "criterion_id": "C1",
            "evidence_refs": "E1",
            "observed_result": "Observed.",
            "missing_requirement": None,
            "status": "MET",
        },
        "action": "FINAL",
        "final_answer": "Complete answer.",
        "unmet_success_criteria": [],
    })
    assert decision.status == "COMPLETED"
    assert decision.criterion_reviews[0].evidence_refs == ["E1"]


def test_scalar_sequence_repair_is_shared_across_worker_schemas():
    general = GeneralResult.model_validate({
        "summary": "Observed result.",
        "unresolved_items": 7,
        "evidence_tool_call_ids": "E1",
    })
    assert general.unresolved_items == ["7"]
    assert general.evidence_tool_call_ids == ["E1"]

    web = WorkerSubmission.model_validate({
        "summary": "Observed web facts.",
        "final_conclusion": "Observed.",
        "stop_reason": "Evidence collected.",
        "criterion_claims": {
            "criterion_id": "C1",
            "criterion": "Check the source.",
            "evidence_tool_call_ids": "WA1",
            "conclusion": "Observed.",
            "status": "MET",
        },
        "confirmed_results": 9,
    })
    assert web.confirmed_results == ["9"]
    assert web.criterion_claims[0].evidence_tool_call_ids == ["WA1"]

    code = CodeWorkerSubmission.model_validate({
        "candidate": CodeCandidateRef(
            event_id="evt", step_id=1, attempt_id="a1",
            workspace_id="w1", candidate_revision=1,
        ),
        "summary": "Implemented.",
        "requirement_status": {"r1": "MET"},
        "limitations": 3,
        "evidence_tool_call_ids": "E2",
    })
    assert code.limitations == ("3",)
    assert code.evidence_tool_call_ids == ("E2",)

    finding = CodeReviewFinding.model_validate({
        "finding_id": "F1",
        "summary": "Observed defect.",
        "affected_requirement_ids": "r1",
    })
    assert finding.affected_requirement_ids == ("r1",)


def test_high_volume_report_fields_keep_generous_but_finite_limits():
    step = _properties(StepReport)
    assert step["summary"]["maxLength"] == 8000
    assert step["answer_text"]["anyOf"][0]["maxLength"] == 32000
    assert step["criterion_results"]["maxItems"] == 24
    assert step["evidence"]["maxItems"] == 48

    final = _properties(FinalReviewDecision)
    assert final["review_reason"]["maxLength"] == 8000
    assert final["final_answer"]["anyOf"][0]["maxLength"] == 48000
    assert final["criterion_reviews"]["maxItems"] == 24


def test_terminal_report_is_deferred_but_sibling_calls_are_not_discarded():
    runtime = SimpleNamespace(
        state={
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "read", "name": "read_file", "args": {}},
                        {"id": "report", "name": "complete_web_step", "args": {}},
                    ],
                )
            ]
        },
        tool_call_id="report",
    )
    command = defer_terminal_call_if_mixed(runtime, "complete_web_step")
    assert command is not None
    message = command.update["messages"][0]
    assert message.status == "error"
    assert "read_file" in message.content
    assert "submit the report alone" in message.content


def test_independent_reporter_reads_materials_and_defers_same_round_report():
    class Model:
        def bind_tools(self, tools):
            self.reader_name = tools[0].name
            return self

        async def ainvoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "read-1",
                        "name": self.reader_name,
                        "args": {"requests": [{"reference": "M404", "offset": 0}]},
                    },
                    {"id": "report-1", "name": "StepReport", "args": {}},
                ],
            )

    messages = []
    runner = ReviewWithReads(
        Model(), Model(), SimpleNamespace(attempts=[]), rounds=5
    )
    result = asyncio.run(runner.ainvoke(messages))
    assert result == {"material_read": True}
    tool_messages = [item for item in messages if isinstance(item, ToolMessage)]
    assert {item.tool_call_id for item in tool_messages} == {"read-1", "report-1"}
    assert next(item for item in tool_messages if item.tool_call_id == "report-1").status == "error"
