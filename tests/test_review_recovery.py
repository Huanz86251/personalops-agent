import unittest

from planning_models import StepReport
from review_recovery import (
    CodeRecoveryProposal,
    GeneralRecoveryProposal,
    WebRecoveryProposal,
    build_recovery_proposal,
)
from scheduler_runtime import scheduler_report_projection


class _StructuredModel:
    def __init__(self, payloads):
        self.payloads = payloads
        self.schema = None
        self.messages = None

    def with_structured_output(self, schema, **_kwargs):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        payload = self.payloads[self.schema.__name__]
        return {
            "parsed": self.schema.model_validate(payload),
            "parsing_error": None,
        }


class ReviewRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_role_specific_proposals_use_distinct_schemas_and_prompts(self):
        model = _StructuredModel({
            "WebRecoveryProposal": {
                "proposal_kind": "WEB",
                "affected_criterion_ids": ["C1"],
                "unavailable_or_uncertain_facts": ["Current popularity is unverified."],
                "attempted_approaches": ["Several public searches were reviewed."],
                "strongest_supported_claim": "A currently available recommendation.",
                "reason": "No direct ranking source was verified.",
                "preserved_results": ["The recipe itself was checked."],
                "downgrade_recommended": True,
                "downgrade_reason": "Popularity could not be ranked.",
                "proposed_degraded_outcome": "Return one verified available option.",
                "user_visible_disclosure": "Current viral status was not independently verified.",
                "user_confirmation_request": "Popularity could not be verified. Allow one verified available option instead?",
            },
            "GeneralRecoveryProposal": {
                "proposal_kind": "GENERAL",
                "affected_criterion_ids": ["C2"],
                "blocked_requirements": ["The requested write scope is unavailable."],
                "external_state_impact": "No external state would be changed.",
                "preserved_safety_boundaries": ["Do not widen the target set."],
                "reason": "The requested mutation cannot be verified.",
                "preserved_results": ["The target records were read."],
                "downgrade_recommended": True,
                "downgrade_reason": "The write effect could not be verified.",
                "proposed_degraded_outcome": "Return the verified read-only result.",
                "user_visible_disclosure": "No records were changed.",
                "user_confirmation_request": "No records can be safely changed. Allow a read-only result instead?",
            },
            "CodeRecoveryProposal": {
                "proposal_kind": "CODE",
                "affected_requirement_ids": ["R2"],
                "candidate_revision": "rev-2",
                "verified_files": ["app.py"],
                "verified_requirements": ["R1"],
                "unmet_requirements": ["R2"],
                "retained_interfaces": ["cli"],
                "validation_required_after_change": ["Run the CLI test suite."],
                "reason": "The optional backend cannot be implemented in this runtime.",
                "preserved_results": ["The CLI path passed."],
                "downgrade_recommended": True,
                "downgrade_reason": "The optional backend has no credible implementation path.",
                "proposed_degraded_outcome": "Deliver only the verified CLI path.",
                "user_visible_disclosure": "The optional backend remains unsupported.",
                "user_confirmation_request": "The CLI path passed but the optional backend did not. Allow CLI-only delivery?",
            },
        })

        for role, expected in (
            ("GENERAL", "GENERAL结果审核"),
            ("CODE", "CODE候选审核"),
        ):
            value, calls, error = await build_recovery_proposal(
                model,
                worker_kind=role,
                user_request="Do the task.",
                current_step={"step_id": 1, "worker_kind": role},
                review_output={"status": "PARTIAL"},
                trigger_source="test",
            )
            self.assertEqual(calls, 1)
            self.assertIsNone(error)
            self.assertEqual(value["proposal_kind"], role)
            self.assertIn(expected, model.messages[0]["content"])

        value, calls, error = await build_recovery_proposal(
            model,
            worker_kind="WEB",
            user_request="Do the task.",
            current_step={"step_id": 1, "worker_kind": "WEB"},
            review_output={"status": "PARTIAL"},
            trigger_source="test",
        )
        self.assertIsNone(value)
        self.assertEqual(calls, 0)
        self.assertIn("Unsupported recovery proposal role: WEB", error)

    def test_no_downgrade_forbids_confirmation_fields(self):
        with self.assertRaises(ValueError):
            WebRecoveryProposal.model_validate({
                "proposal_kind": "WEB",
                "reason": "A different source may still satisfy the original request.",
                "preserved_results": [],
                "downgrade_recommended": False,
                "downgrade_reason": None,
                "proposed_degraded_outcome": None,
                "user_visible_disclosure": None,
                "user_confirmation_request": "May I weaken the request?",
                "affected_criterion_ids": ["C1"],
                "unavailable_or_uncertain_facts": ["One source failed."],
                "attempted_approaches": ["First source."],
                "strongest_supported_claim": "No claim yet.",
            })

    def test_all_reviewer_roles_can_recommend_exact_downgrade(self):
        common = {
            "reason": "The frozen contract was not satisfied.",
            "preserved_results": ["One result was verified."],
            "downgrade_recommended": True,
            "downgrade_reason": "One material requirement remains unavailable.",
            "proposed_degraded_outcome": "Deliver the verified subset only.",
            "user_visible_disclosure": "One original requirement will be omitted.",
            "user_confirmation_request": "May I deliver only the verified subset?",
        }
        values = [
            WebRecoveryProposal.model_validate({
                **common,
                "proposal_kind": "WEB",
                "affected_criterion_ids": ["C1"],
                "unavailable_or_uncertain_facts": ["Ranking unavailable."],
                "attempted_approaches": [],
                "strongest_supported_claim": "One verified option.",
            }),
            GeneralRecoveryProposal.model_validate({
                **common,
                "proposal_kind": "GENERAL",
                "affected_criterion_ids": ["C1"],
                "blocked_requirements": ["Requested output format."],
                "external_state_impact": "No external mutation.",
                "preserved_safety_boundaries": [],
            }),
            CodeRecoveryProposal.model_validate({
                **common,
                "proposal_kind": "CODE",
                "affected_requirement_ids": ["R1"],
                "candidate_revision": "rev-1",
                "verified_files": ["a.py"],
                "verified_requirements": [],
                "unmet_requirements": ["R1"],
                "retained_interfaces": [],
                "validation_required_after_change": ["Re-run tests."],
            }),
        ]
        self.assertEqual([value.proposal_kind for value in values], ["WEB", "GENERAL", "CODE"])

    def test_scheduler_projection_keeps_verdict_and_drops_worker_context(self):
        report = StepReport(
            step_id=1,
            summary="Partially verified.",
            stop_reason="Evidence gap.",
            status="PARTIAL",
            answer_text="full worker draft",
            handoff_knowledge=[],
            handoff_apis=[],
            recovery_proposal={"proposal_kind": "WEB"},
            request_replan=True,
            replan_reason="Need a recovery decision.",
        )

        projected = scheduler_report_projection(report)

        self.assertEqual(projected["recovery_proposal"]["proposal_kind"], "WEB")
        self.assertNotIn("answer_text", projected)
        self.assertNotIn("handoff_knowledge", projected)
        self.assertNotIn("handoff_apis", projected)
        self.assertNotIn("evidence", projected)

    def test_first_pass_report_schema_hides_harness_recovery_field(self):
        self.assertNotIn("recovery_proposal", StepReport.model_json_schema()["properties"])


if __name__ == "__main__":
    unittest.main()
