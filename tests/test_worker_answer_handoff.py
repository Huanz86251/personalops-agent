"""Text deliverables must survive Worker -> review -> Final Reviewer handoff."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from planning_graph import _build_code_step_report, _build_general_step_report
from planning_models import PlanStep, StepReport
from reporting import build_step_review_packet, materialize_worker_review_trace
from scheduler_runtime import compact_json
from workers.code_review_models import (
    CodeCandidateRef,
    CodeReviewLoopState,
    CodeReviewReport,
    CodeWorkerSubmission,
)
from workers.general_completion import GeneralResult
from workers.submission import WorkerSubmission, WorkerSubmissionRecord
from workers.evidence_refs import (
    EvidenceReferenceMiddleware,
    MIGRATED_FINAL_CONCLUSION,
    migrate_oversized_final_conclusion,
)
from workers.submission import complete_web_step


ANSWER = "**第一点**：补充项目成果。\n\n**第二点**：核对日期与证据。"


def test_general_text_is_separate_from_summary_and_keeps_layout():
    result = GeneralResult(status="COMPLETED", summary="已阅读简历。", answer_text=ANSWER)
    assert "answer_text" in GeneralResult.model_json_schema()["properties"]
    report = _build_general_step_report(
        PlanStep(step_id=1, objective="提出简历建议", success_criteria=["给出具体建议"]),
        {"general_result": result.model_dump(mode="json"), "finish_reason": "NATURAL_EXIT"},
        SimpleNamespace(attempts=[]),
    )
    assert report.summary == "已阅读简历。"
    assert report.answer_text == ANSWER
    assert json.loads(compact_json({"step_reports": [report]}))["step_reports"][0]["answer_text"] == ANSWER
    assert "answer_text" in StepReport.model_json_schema()["properties"]


def test_web_text_survives_bounded_review_packet_without_collapsing_paragraphs():
    record = WorkerSubmissionRecord(
        submitted_at=datetime.now(timezone.utc),
        submission=WorkerSubmission(
            summary="已查证资料。",
            final_conclusion="建议草稿待审核。",
            answer_text=ANSWER,
        ),
        total_tool_calls=0,
    )
    trace = {"attempt": 1, "worker_submission": record.model_dump(mode="json")}
    durable = materialize_worker_review_trace(trace)
    packet = build_step_review_packet(
        user_request="分析我的简历",
        plan_objective="提出建议",
        current_step=PlanStep(step_id=1, worker_kind="WEB", objective="核对简历", success_criteria=["给出建议"]),
        current_attempt=durable,
        stop_reason="Worker submitted.",
    )
    assert packet.attempts[0].answer_text == ANSWER
    assert packet.attempts[0].summary == "已查证资料。"
    assert "answer_text" in WorkerSubmission.model_json_schema()["properties"]


def test_code_text_reaches_step_report_even_when_review_fails():
    candidate = CodeCandidateRef(
        event_id="event", step_id=1, attempt_id="attempt", workspace_id="workspace", candidate_revision=1
    )
    submission = CodeWorkerSubmission(
        candidate=candidate,
        summary="实现并检查完成。",
        answer_text=ANSWER,
        requirement_status={"R1": "MET"},
    )
    assert "answer_text" in CodeWorkerSubmission.model_json_schema()["properties"]
    loop = CodeReviewLoopState(
        candidate=candidate,
        worker_checkpoint_id="worker-checkpoint",
        reviewer_checkpoint_id="reviewer-checkpoint",
        status="STOPPED",
        terminal_summary="未通过独立审核。",
    )
    review = CodeReviewReport(
        candidate=candidate,
        summary="仍有未解决项。",
        verification_summary="缺少一项检查。",
        verdict="FAILED",

    )
    report = _build_code_step_report(
        PlanStep(
            step_id=1, objective="交付说明", worker_kind="CODE",
            success_criteria=["交付说明"],
        ),
        {
            "code_worker_submission": {"submission": submission.model_dump(mode="json")},
            "code_review_loop": loop.model_dump(mode="json"),
            "code_review_report": review.model_dump(mode="json"),
        },
    )
    assert report.answer_text == ANSWER
    assert report.status != "COMPLETED"


def test_submission_prompt_explains_text_field_positive_and_negative_examples():
    prompt = complete_web_step.description
    assert "完整的分点建议正文" in prompt
    assert "整篇建议塞进" in prompt
    assert "answer_text=null" in prompt


def test_final_conclusion_overflow_feedback_moves_body_to_answer_text():
    request = SimpleNamespace(
        state={},
        tool_call={"name": "complete_web_step", "id": "submit-1"},
        tool=complete_web_step,
    )
    result = EvidenceReferenceMiddleware()._error(
        request,
        "submission.final_conclusion: String should have at most 1600 characters",
    )
    message = result.update["messages"][0]
    assert "submission.answer_text" in message.content
    assert "final_conclusion缩成" in message.content
    assert "不要重新调用业务API" in message.content


def test_final_conclusion_overflow_is_repaired_before_tool_validation():
    body = "完整正文" * 500
    repaired = migrate_oversized_final_conclusion({
        "submission": {
            "summary": "已完成",
            "answer_text": None,
            "final_conclusion": body,
        }
    })
    assert repaired["submission"]["answer_text"] == body
    assert repaired["submission"]["final_conclusion"] == MIGRATED_FINAL_CONCLUSION


def test_unmigratable_oversize_body_is_not_silently_truncated():
    body = "x" * 32001
    original = {"submission": {"answer_text": None, "final_conclusion": body}}
    assert migrate_oversized_final_conclusion(original) == original
