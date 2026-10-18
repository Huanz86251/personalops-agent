"""Offline tests for the Scheduler/Worker/Reviewer CODE protocol."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from planning_models import (
    CodeInterfaceRequirement,
    CodeRequirement,
    CodeTaskContract,
    PlanStep,
)
from workers.code_review_models import (
    CodeCandidateRef,
    CodeChangedFile,
    CodeCheckResult,
    CodeRepairInstruction,
    CodeReviewFinding,
    CodeReviewReport,
    CodeWorkerRepairResponse,
    CodeWorkerRepairPayload,

    create_code_review_loop,
    finish_code_review,
    record_code_publication,
    receive_code_worker_response,

    request_code_repair,
)


def candidate(revision: int = 1) -> CodeCandidateRef:
    return CodeCandidateRef(
        event_id="event-1",
        step_id=1,
        attempt_id="attempt-1",
        workspace_id="workspace-1",
        candidate_revision=revision,
    )


def finding() -> CodeReviewFinding:
    return CodeReviewFinding(
        finding_id="route_returns_500",
        summary="The health route returns HTTP 500.",
        affected_requirement_ids=("health_route",),
    )


class CodeSchedulerContractTests(unittest.TestCase):
    def test_finding_schema_has_no_category(self) -> None:
        schema = CodeReviewFinding.model_json_schema()
        self.assertNotIn("category", schema["properties"])
        self.assertNotIn("category", schema.get("required", []))
        self.assertEqual(finding().summary, "The health route returns HTTP 500.")

    def test_code_step_requires_a_frozen_contract(self) -> None:
        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                objective="Implement a health route.",
                success_criteria=[],
                worker_kind="CODE",
            )

        step = PlanStep(
            step_id=1,
            objective="Implement a health route.",
            success_criteria=["GET /health returns 200."],
            worker_kind="CODE",
            code_task=CodeTaskContract(
                requirements=[
                    CodeRequirement(
                        requirement_id="health_route",
                        priority="MUST",
                        statement="GET /health returns HTTP 200.",
                    ),
                    CodeRequirement(
                        requirement_id="small_patch",
                        priority="SHOULD",
                        statement="Keep the patch focused.",
                    ),
                ],
                interfaces=[
                    CodeInterfaceRequirement(
                        interface_id="health_http",
                        kind="HTTP_API",
                        description="Service health endpoint.",
                        details={"method": "GET", "path": "/health"},
                    )
                ],
                validation_expectations=["Call the route and assert HTTP 200."],
                non_goals=["Do not redesign authentication."],
            ),
        )

        self.assertEqual(step.code_task.requirements[0].priority, "MUST")
        self.assertNotIn("implementation_guidance", step.code_task.model_dump())
        self.assertNotIn("implementation_guidance", CodeTaskContract.model_json_schema()["properties"])
        self.assertEqual(step.code_task.interfaces[0].details["path"], "/health")

    def test_code_criteria_migrate_only_when_requirements_missing(self):
        base = dict(step_id=1, objective="Health endpoint", worker_kind="CODE")
        for contract in (None, {}, {"requirements": [], "validation_expectations": ["Call the route"]}):
            step = PlanStep(**base, success_criteria=["HTTP 200", "JSON response"], code_task=contract)
            self.assertEqual(step.success_criteria, [])
            self.assertEqual([r.requirement_id for r in step.code_task.requirements], ["criterion_1", "criterion_2"])
            self.assertEqual([r.priority for r in step.code_task.requirements], ["MUST", "MUST"])
            self.assertEqual(step.acceptance_criteria, ["HTTP 200", "JSON response"])
            self.assertEqual(PlanStep.model_validate(step.model_dump()).model_dump(), step.model_dump())
        step = PlanStep(**base, success_criteria=["Discard duplicate"], code_task={"requirements": [
            {"requirement_id": "health", "statement": "Authoritative requirement"},
            {"requirement_id": "style", "priority": "SHOULD", "statement": "Preference"}]})
        self.assertEqual(step.success_criteria, [])
        self.assertEqual(step.acceptance_criteria, ["Authoritative requirement"])
        self.assertEqual(len(step.code_task.requirements), 2)
        with self.assertRaises(ValidationError):
            PlanStep(**base, success_criteria=[], code_task=None)
        with self.assertRaises(ValidationError):
            PlanStep(step_id=1, objective="Search", worker_kind="WEB", success_criteria=[])

    def test_code_requirements_reach_step_review_and_final_review(self):
        from hard_planning import _final_review_step_projection
        from reporting.context import build_step_review_packet
        step = PlanStep(step_id=1, objective="Health endpoint", worker_kind="CODE",
                        success_criteria=["HTTP 200"])
        packet = build_step_review_packet(user_request="Health endpoint", plan_objective="Health endpoint",
                                         current_step=step, current_attempt={"worker_id": "code-worker"},
                                         stop_reason="Offline fixture")
        self.assertEqual(packet.task_contract.success_criteria, ["HTTP 200"])
        projected = _final_review_step_projection(step)
        self.assertEqual(projected["success_criteria"], [])
        self.assertEqual(projected["code_task"]["requirements"][0]["statement"], "HTTP 200")

    def test_non_code_step_rejects_code_contract(self) -> None:
        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                objective="Research a page.",
                success_criteria=["Return a cited result."],
                worker_kind="WEB",
                code_task=CodeTaskContract(
                    requirements=[
                        CodeRequirement(
                            requirement_id="wrong_scope",
                            statement="This contract is misplaced.",
                        )
                    ]
                ),
            )


class CodeReviewLoopTests(unittest.TestCase):
    def test_repair_tool_binds_round_and_preserves_checkpoint_response(self) -> None:
        from workers.code_submission import respond_to_code_review
        state = create_code_review_loop(candidate=candidate(),
            worker_checkpoint_id="w", reviewer_checkpoint_id="r", max_repair_rounds=3)
        for round_no in (1, 2):
            waiting = request_code_repair(state, summary="Check test expectations.",
                required_changes=("Inspect the test.",), findings=(finding(),))
            runtime = SimpleNamespace(state={"code_review_loop": waiting.model_dump(mode="json")},
                tool_call_id="repair", stream_writer=lambda event: None)
            result = respond_to_code_review.func(response=CodeWorkerRepairPayload(
                action="TEST_DISPUTE", candidate=candidate(), summary="Test expects wrong output."),
                updated_submission=None, runtime=runtime)
            self.assertEqual(result.update["code_worker_repair_response"]["round_no"], round_no)
            state = type(waiting).model_validate(result.update["code_review_loop"])
            self.assertEqual(state.exchanges[-1].response.round_no, round_no)
            self.assertIsNone(state.pending_instruction)
            restored = type(state).model_validate_json(state.model_dump_json())
            self.assertEqual(restored.exchanges[-1].response.round_no, round_no)

    def test_repair_tool_does_not_accept_response_without_pending_instruction(self) -> None:
        from workers.code_submission import respond_to_code_review
        loop = create_code_review_loop(candidate=candidate(),
            worker_checkpoint_id="w", reviewer_checkpoint_id="r")
        runtime = SimpleNamespace(state={"code_review_loop": loop.model_dump(mode="json")},
            tool_call_id="repair", stream_writer=lambda event: None)
        result = respond_to_code_review.func(response=CodeWorkerRepairPayload(
            action="TEST_DISPUTE", candidate=candidate(), summary="No repair is pending."),
            updated_submission=None, runtime=runtime)
        self.assertNotIn("code_worker_repair_response", result.update)
        self.assertIn("rejected", result.update["messages"][0].content)

    def test_repair_tool_schema_hides_round_number(self) -> None:
        from workers.code_submission import respond_to_code_review
        schema = respond_to_code_review.tool_call_schema.model_json_schema()
        payload = schema["$defs"]["CodeWorkerRepairPayload"]
        self.assertNotIn("round_no", payload["properties"])
        self.assertIn("round_no", CodeWorkerRepairResponse.model_json_schema()["properties"])

    def test_revision_round_preserves_pair_and_history(self) -> None:
        state = create_code_review_loop(
            candidate=candidate(),
            worker_checkpoint_id="worker-thread-1",
            reviewer_checkpoint_id="reviewer-thread-1",
            max_repair_rounds=2,
        )
        waiting = request_code_repair(
            state,
            summary="The route is broken.",
            required_changes=("Fix the health route.",),
            preserve_behaviors=("Keep the existing API stable.",),
            findings=(finding(),),
        )
        reviewing = receive_code_worker_response(
            waiting,
            CodeWorkerRepairResponse(
                round_no=1,
                action="REVISION_READY",
                candidate=candidate(2),
                summary="Fixed the route.",
                changed_files=(
                    CodeChangedFile(
                        path="app/routes.py",
                        change_summary="Return a health response.",
                    ),
                ),
            ),
        )

        self.assertEqual(reviewing.status, "REVIEWING")
        self.assertEqual(reviewing.candidate.candidate_revision, 2)
        self.assertEqual(reviewing.worker_checkpoint_id, "worker-thread-1")
        self.assertEqual(reviewing.reviewer_checkpoint_id, "reviewer-thread-1")
        self.assertEqual(len(reviewing.exchanges), 1)

    def test_stale_or_switched_candidate_is_rejected(self) -> None:
        state = request_code_repair(
            create_code_review_loop(
                candidate=candidate(),
                worker_checkpoint_id="worker-thread-1",
                reviewer_checkpoint_id="reviewer-thread-1",
            ),
            summary="Fix it.",
            required_changes=("Fix it.",),
            findings=(finding(),),
        )
        switched = candidate(2).model_copy(update={"attempt_id": "attempt-2"})

        with self.assertRaises(ValueError):
            receive_code_worker_response(
                state,
                CodeWorkerRepairResponse(
                    round_no=1,
                    action="REVISION_READY",
                    candidate=switched,
                    summary="Changed a different attempt.",
                ),
            )


    def test_published_report_closes_without_scheduler_accept(self) -> None:
        state = create_code_review_loop(
            candidate=candidate(),
            worker_checkpoint_id="worker-thread-1",
            reviewer_checkpoint_id="reviewer-thread-1",
        )
        published = record_code_publication(
            state,
            candidate=candidate(),
        )
        applied = finish_code_review(
            published,
            CodeReviewReport(
                candidate=candidate(),
                verdict="PASSED",
                summary="The implementation passed focused checks.",
                verification_summary="The health route check passed.",
                verified_requirement_ids=("health_route",),
                check_results=(
                    CodeCheckResult(
                        check_id="health-route",
                        description="Call GET /health.",
                        status="PASSED",
                        summary="The route returned HTTP 200.",
                    ),
                ),
                approved_artifact_paths=("app/routes.py",),
                published_artifact_paths=("app/routes.py",),
                delivery_location="D:/PythonProject",
                publication_id="publication-1",
                applied_revision="sha256:applied",
            ),
        )

        self.assertEqual(applied.status, "APPLIED")
        self.assertEqual(
            applied.terminal_summary,
            "The implementation passed focused checks.",
        )


if __name__ == "__main__":
    unittest.main()
