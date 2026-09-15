"""Offline contracts for single and parallel PlanStep shapes."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from planning_models import PlanStep, StepArtifactOutput, WorkerAssignment


def web_assignment(key: str) -> WorkerAssignment:
    return WorkerAssignment(
        assignment_key=key,
        objective=f"Research the {key} direction and return cited findings.",
    )


class ParallelPlanModelTests(unittest.TestCase):
    def test_single_step_remains_the_cheap_default(self):
        step = PlanStep(
            step_id=1,
            objective="Implement and verify one local change.",
            success_criteria=["The local change passes its focused test."],
        )

        self.assertEqual(step.execution_mode, "SINGLE")
        self.assertEqual(step.worker_assignments, [])
        self.assertEqual(step.join_policy, "ALL_TERMINAL")

    def test_parallel_web_step_accepts_two_or_three_unique_branches(self):
        step = PlanStep(
            step_id=1,
            objective="Compare current cinema options.",
            success_criteria=["Cover official listings and independent reviews."],
            worker_kind="WEB",
            execution_mode="PARALLEL",
            worker_assignments=[
                web_assignment("official_sources"),
                web_assignment("independent_reviews"),
                web_assignment("price_crosscheck"),
            ],
        )

        self.assertEqual(len(step.worker_assignments), 3)
        self.assertEqual(step.join_policy, "ALL_TERMINAL")

    def test_parallel_step_requires_at_least_two_branches(self):
        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                objective="Research one topic.",
                success_criteria=["Return evidence."],
                worker_kind="WEB",
                execution_mode="PARALLEL",
                worker_assignments=[web_assignment("only_branch")],
            )

    def test_parallel_step_rejects_duplicate_assignment_keys(self):
        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                objective="Research two sources.",
                success_criteria=["Return both sources."],
                worker_kind="WEB",
                execution_mode="PARALLEL",
                worker_assignments=[
                    web_assignment("source"),
                    web_assignment("source"),
                ],
            )

    def test_v1_rejects_parallel_code_or_general_steps(self):
        for worker_kind in ("CODE", "GENERAL"):
            with self.subTest(worker_kind=worker_kind):
                with self.assertRaises(ValidationError):
                    PlanStep(
                        step_id=1,
                        objective="Run parallel work.",
                        success_criteria=["Return both results."],
                        worker_kind=worker_kind,
                        execution_mode="PARALLEL",
                        worker_assignments=[
                            WorkerAssignment(
                                assignment_key="first",
                                objective="First branch.",
                            ),
                            web_assignment("second"),
                        ],
                    )

    def test_single_step_rejects_redundant_assignments(self):
        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                objective="Do one task.",
                success_criteria=["Return one result."],
                worker_assignments=[web_assignment("redundant")],
            )

    def test_web_step_declares_internal_and_user_artifacts_explicitly(self):
        step = PlanStep(
            step_id=1,
            objective="Download a reference and a requested HTML file.",
            success_criteria=["Both files are verified."],
            worker_kind="WEB",
            artifact_outputs=[
                StepArtifactOutput(
                    output_id="reference_notes",
                    description="Notes for a later Code Step.",
                ),
                StepArtifactOutput(
                    output_id="requested_html",
                    description="The HTML file requested by the user.",
                    disposition="USER_DELIVERABLE",
                    target_path="downloads/template.html",
                ),
            ],
        )

        self.assertEqual(
            step.artifact_outputs[0].disposition,
            "INTERNAL_HANDOFF",
        )
        self.assertEqual(
            step.artifact_outputs[1].target_path,
            "downloads/template.html",
        )

    def test_user_deliverable_requires_safe_relative_target(self):
        for target in (None, "../escape.html", "C:/escape.html", ".git/config"):
            with self.subTest(target=target):
                with self.assertRaises(ValidationError):
                    StepArtifactOutput(
                        output_id="requested_html",
                        description="Requested HTML.",
                        disposition="USER_DELIVERABLE",
                        target_path=target,
                    )

    def test_code_step_cannot_mix_artifact_contracts(self):
        from planning_models import CodeRequirement, CodeTaskContract

        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                objective="Write code.",
                success_criteria=["Code is complete."],
                worker_kind="CODE",
                code_task=CodeTaskContract(
                    requirements=[
                        CodeRequirement(
                            requirement_id="feature",
                            statement="Implement the feature.",
                        )
                    ]
                ),
                artifact_outputs=[
                    StepArtifactOutput(
                        output_id="duplicate_contract",
                        description="Should be rejected.",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
