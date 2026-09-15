"""Offline tests for the Scheduler/Worker/Reviewer CODE protocol."""

from __future__ import annotations

import unittest

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
    CodeContinuationSubmission,
    SchedulerCodeDecision,
    apply_scheduler_code_decision,
    create_code_review_loop,
    finish_code_review,
    record_code_publication,
    receive_code_worker_response,
    receive_code_continuation_submission,
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
        category="CANDIDATE_DEFECT",
        summary="The health route returns HTTP 500.",
        affected_requirement_ids=("health_route",),
    )


class CodeSchedulerContractTests(unittest.TestCase):
    def test_code_step_requires_a_frozen_contract(self) -> None:
        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                objective="Implement a health route.",
                success_criteria=["GET /health returns 200."],
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
                implementation_guidance=["Reuse the existing router."],
                non_goals=["Do not redesign authentication."],
            ),
        )

        self.assertEqual(step.code_task.requirements[0].priority, "MUST")
        self.assertEqual(step.code_task.interfaces[0].details["path"], "/health")

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

    def test_scheduler_continue_reuses_context_and_resets_local_budget(self) -> None:
        state = request_code_repair(
            create_code_review_loop(
                candidate=candidate(),
                worker_checkpoint_id="worker-thread-1",
                reviewer_checkpoint_id="reviewer-thread-1",
                max_repair_rounds=1,
            ),
            summary="Fix it.",
            required_changes=("Fix it.",),
            findings=(finding(),),
        )
        reviewing_blocker = receive_code_worker_response(
            state,
            CodeWorkerRepairResponse(
                round_no=1,
                action="BLOCKED",
                candidate=candidate(),
                summary="The requirement needs a stronger decision.",
                reasons=("Two incompatible routes already exist.",),
            ),
        )
        escalated = finish_code_review(
            reviewing_blocker,
            CodeReviewReport(
                candidate=candidate(),
                verdict="ESCALATED",
                summary="The requirement needs Scheduler guidance.",
                verification_summary=(
                    "Reviewer confirmed the ambiguity before publication."
                ),
                recommended_action="CONTINUE",
            ),
        )
        continued = apply_scheduler_code_decision(
            escalated,
            SchedulerCodeDecision(
                action="CONTINUE",
                reason="Use the public route.",
                worker_instruction="Implement the public route only.",
                reviewer_instruction="Verify the public route only.",
                repair_rounds=2,
            ),
        )

        self.assertEqual(continued.status, "WAITING_FOR_WORKER")
        self.assertEqual(continued.scheduler_epoch, 2)
        self.assertEqual(continued.repair_round, 0)
        self.assertEqual(continued.max_repair_rounds, 2)
        self.assertIsNotNone(continued.pending_scheduler_directive)
        self.assertEqual(
            continued.pending_scheduler_directive.worker_instruction,
            "Implement the public route only.",
        )
        self.assertEqual(
            continued.pending_scheduler_directive.reviewer_instruction,
            "Verify the public route only.",
        )
        self.assertEqual(len(continued.exchanges), 1)
        self.assertEqual(continued.worker_checkpoint_id, "worker-thread-1")
        self.assertEqual(continued.reviewer_checkpoint_id, "reviewer-thread-1")

        reviewing = receive_code_continuation_submission(
            continued,
            CodeContinuationSubmission(
                scheduler_epoch=2,
                action="REVISION_READY",
                candidate=candidate(2),
                summary="Implemented the Scheduler's clarified route.",
            ),
        )

        self.assertEqual(reviewing.status, "REVIEWING")
        self.assertEqual(reviewing.candidate.candidate_revision, 2)
        self.assertIsNone(reviewing.pending_scheduler_directive)
        self.assertEqual(len(reviewing.scheduler_continuations), 1)
        self.assertEqual(
            reviewing.scheduler_continuations[0].directive.reviewer_instruction,
            "Verify the public route only.",
        )
        self.assertEqual(reviewing.worker_checkpoint_id, "worker-thread-1")
        self.assertEqual(reviewing.reviewer_checkpoint_id, "reviewer-thread-1")

    def test_scheduler_continue_rejects_stale_epoch(self) -> None:
        state = create_code_review_loop(
            candidate=candidate(),
            worker_checkpoint_id="worker-thread-1",
            reviewer_checkpoint_id="reviewer-thread-1",
        ).model_copy(
            update={
                "status": "ESCALATED_TO_SCHEDULER",
                "terminal_summary": "Needs Scheduler guidance.",
            }
        )
        continued = apply_scheduler_code_decision(
            state,
            SchedulerCodeDecision(
                action="CONTINUE",
                reason="Keep the existing API.",
                worker_instruction="Repair the existing API.",
                reviewer_instruction="Retest the existing API.",
                repair_rounds=2,
            ),
        )

        with self.assertRaises(ValueError):
            receive_code_continuation_submission(
                continued,
                CodeContinuationSubmission(
                    scheduler_epoch=1,
                    action="REVISION_READY",
                    candidate=candidate(2),
                    summary="This response belongs to an old epoch.",
                ),
            )

    def test_continuation_blocker_still_routes_to_reviewer(self) -> None:
        state = create_code_review_loop(
            candidate=candidate(),
            worker_checkpoint_id="worker-thread-1",
            reviewer_checkpoint_id="reviewer-thread-1",
        ).model_copy(
            update={
                "status": "ESCALATED_TO_SCHEDULER",
                "terminal_summary": "Needs Scheduler guidance.",
            }
        )
        continued = apply_scheduler_code_decision(
            state,
            SchedulerCodeDecision(
                action="CONTINUE",
                reason="Try the clarified interface.",
                worker_instruction="Implement the clarified interface.",
                reviewer_instruction="Assess the implementation or blocker.",
                repair_rounds=2,
            ),
        )

        reviewing = receive_code_continuation_submission(
            continued,
            CodeContinuationSubmission(
                scheduler_epoch=2,
                action="BLOCKED",
                candidate=candidate(),
                summary="The interface is unavailable in this environment.",
                reasons=("Required service is missing.",),
            ),
        )

        self.assertEqual(reviewing.status, "REVIEWING")
        self.assertIsNone(reviewing.terminal_summary)
        self.assertEqual(
            reviewing.scheduler_continuations[-1].submission.action,
            "BLOCKED",
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
