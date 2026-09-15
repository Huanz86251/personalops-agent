"""Provider-free integration tests for dedicated CODE Deep Agents."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from planning_models import CodeRequirement, CodeTaskContract
from prompt_loader import load_prompt
from workers.code_reviewer import (
    CODE_REVIEWER_FILESYSTEM_TOOLS,
    create_code_reviewer,
)
from workers.code_publisher import PUBLISH_REVIEWED_CANDIDATE_NAME
from workers.code_review_models import (
    CodeCandidateRef,
    CodeReviewLoopState,
    SchedulerCodeDecision,
    apply_scheduler_code_decision,
    create_code_review_loop,
    record_code_publication,
)
from workers.code_submission import (
    REQUEST_CODE_WORKER_REPAIR_NAME,
    SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME,
    SUBMIT_CODE_REVIEW_NAME,
    SUBMIT_CODE_FOR_REVIEW_NAME,
)
from workers.code_worker import (
    CODE_WORKER_FILESYSTEM_TOOLS,
    create_code_worker,
)


@tool
def code_probe(value: str) -> str:
    """Return deterministic evidence for CODE Agent tests."""

    return f"checked:{value}"


def candidate(revision: int = 1) -> CodeCandidateRef:
    return CodeCandidateRef(
        event_id="event-code-1",
        step_id=1,
        attempt_id="attempt-code-1",
        workspace_id="workspace-code-1",
        candidate_revision=revision,
    )


def contract() -> CodeTaskContract:
    return CodeTaskContract(
        requirements=[
            CodeRequirement(
                requirement_id="feature_works",
                statement="The requested feature works.",
            )
        ],
        validation_expectations=["Run one focused check."],
    )


class ToolRecordingModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    invocation_count: int = 0
    bound_tool_names: list[str] = Field(default_factory=list)
    last_request_text: str = ""

    @property
    def _llm_type(self) -> str:
        return "code-agent-script-test"

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [item.name for item in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.last_request_text = "\n".join(
            str(getattr(item, "content", ""))
            for item in messages
        )
        message = self.responses[self.invocation_count]
        self.invocation_count += 1
        return ChatResult(
            generations=[ChatGeneration(message=message)]
        )


class CodeAgentFactoryTests(unittest.TestCase):
    def test_specialized_prompts_are_external_files(self) -> None:
        self.assertIn(
            "submit_code_for_review",
            load_prompt("workers/code_worker"),
        )
        self.assertIn(
            "request_code_worker_repair",
            load_prompt("reviewers/code"),
        )

    def test_worker_submits_a_real_structured_manifest(self) -> None:
        model = ToolRecordingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "code_probe",
                            "args": {"value": "feature"},
                            "id": "probe-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": SUBMIT_CODE_FOR_REVIEW_NAME,
                            "args": {
                                "submission": {
                                    "candidate": candidate().model_dump(
                                        mode="json"
                                    ),
                                    "summary": (
                                        "Implemented the requested feature."
                                    ),
                                    "requirement_status": {
                                        "feature_works": "MET"
                                    },
                                    "implemented_interfaces": [],
                                    "changed_files": [
                                        {
                                            "path": "app.py",
                                            "change_summary": (
                                                "Implemented the feature."
                                            ),
                                        }
                                    ],
                                    "proposed_artifact_paths": [],
                                    "self_checks": [
                                        {
                                            "check": "Focused feature check",
                                            "outcome": "PASSED",
                                            "summary": (
                                                "The probe completed."
                                            ),
                                        }
                                    ],
                                    "evidence_tool_call_ids": ["probe-1"],
                                    "limitations": [],
                                }
                            },
                            "id": "submit-code-1",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        )
        worker = create_code_worker(model, tools=[code_probe])
        result = worker.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Implement it."}
                ],
                "worker_id": "code-worker-1",
                "code_task": contract().model_dump(mode="json"),
                "code_candidate": candidate().model_dump(mode="json"),
            }
        )

        self.assertTrue(result["code_agent_finished"])
        self.assertEqual(
            result["code_worker_submission"]["submission"]["candidate"]
            ["candidate_revision"],
            1,
        )
        self.assertEqual(
            result["code_worker_submission"]["resolved_evidence"][0]
            ["tool_call_id"],
            "probe-1",
        )
        self.assertIn("execute", CODE_WORKER_FILESYSTEM_TOOLS)
        self.assertNotIn("execute", model.bound_tool_names)
        self.assertNotIn("task", model.bound_tool_names)

    def test_same_worker_consumes_scheduler_continue_directive(self) -> None:
        model = ToolRecordingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME,
                            "args": {
                                "submission": {
                                    "scheduler_epoch": 2,
                                    "action": "REVISION_READY",
                                    "candidate": candidate(2).model_dump(
                                        mode="json"
                                    ),
                                    "summary": "Applied the clarified direction.",
                                    "changed_files": [
                                        {
                                            "path": "app.py",
                                            "change_summary": "Updated behavior.",
                                        }
                                    ],
                                    "reasons": [],
                                    "evidence_tool_call_ids": [],
                                },
                                "updated_submission": {
                                    "candidate": candidate(2).model_dump(
                                        mode="json"
                                    ),
                                    "summary": "Updated the requested feature.",
                                    "requirement_status": {
                                        "feature_works": "MET"
                                    },
                                    "implemented_interfaces": [],
                                    "changed_files": [
                                        {
                                            "path": "app.py",
                                            "change_summary": "Updated behavior.",
                                        }
                                    ],
                                    "proposed_artifact_paths": [],
                                    "self_checks": [],
                                    "evidence_tool_call_ids": [],
                                    "limitations": [],
                                },
                            },
                            "id": "scheduler-continue-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        escalated = create_code_review_loop(
            candidate=candidate(),
            worker_checkpoint_id="worker-thread-1",
            reviewer_checkpoint_id="reviewer-thread-1",
        ).model_copy(
            update={
                "status": "ESCALATED_TO_SCHEDULER",
                "terminal_summary": "Needs a Scheduler decision.",
            }
        )
        continued = apply_scheduler_code_decision(
            escalated,
            SchedulerCodeDecision(
                action="CONTINUE",
                reason="Clarify the desired behavior.",
                worker_instruction="Update only the existing behavior.",
                reviewer_instruction="Retest the clarified behavior.",
                repair_rounds=2,
            ),
        )
        worker = create_code_worker(model, tools=[code_probe])

        result = worker.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Continue this CODE step."}
                ],
                "worker_id": "code-worker-1",
                "code_task": contract().model_dump(mode="json"),
                "code_candidate": candidate().model_dump(mode="json"),
                "code_review_loop": continued.model_dump(mode="json"),
                "code_repair_instruction": {
                    "summary": "stale reviewer instruction"
                },
            }
        )

        self.assertTrue(result["code_agent_finished"])
        self.assertEqual(result["code_review_loop"]["status"], "REVIEWING")
        self.assertEqual(
            result["code_candidate"]["candidate_revision"],
            2,
        )
        self.assertEqual(
            len(result["code_review_loop"]["scheduler_continuations"]),
            1,
        )
        self.assertIn(
            SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME,
            model.bound_tool_names,
        )
        self.assertIn(
            "Update only the existing behavior.",
            model.last_request_text,
        )
        self.assertNotIn(
            "Retest the clarified behavior.",
            model.last_request_text,
        )
        self.assertNotIn(
            "stale reviewer instruction",
            model.last_request_text,
        )

        reviewer_model = ToolRecordingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": SUBMIT_CODE_REVIEW_NAME,
                            "args": {
                                "report": {
                                    "candidate": candidate(2).model_dump(
                                        mode="json"
                                    ),
                                    "verdict": "PASSED",
                                    "summary": "The clarified behavior passed.",
                                    "verification_summary": (
                                        "The focused behavior check passed."
                                    ),
                                    "verified_requirement_ids": [
                                        "feature_works"
                                    ],
                                    "verified_interfaces": [],
                                    "check_results": [
                                        {
                                            "check_id": "focused-check",
                                            "description": (
                                                "Check clarified behavior."
                                            ),
                                            "status": "PASSED",
                                            "summary": (
                                                "Observed expected behavior."
                                            ),
                                        }
                                    ],
                                    "failed_test_summaries": [],
                                    "changed_files": [],
                                    "evidence_refs": [],
                                    "approved_artifact_paths": [],
                                    "published_artifact_paths": [],
                                    "delivery_location": "D:/target",
                                    "publication_id": "publication-1",
                                    "applied_revision": "sha256:applied",
                                }
                            },
                            "id": "review-continued-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        reviewer = create_code_reviewer(reviewer_model, tools=[code_probe])
        published_loop = record_code_publication(
            CodeReviewLoopState.model_validate(result["code_review_loop"]),
            candidate=candidate(2),
        )
        published_at = datetime.now(timezone.utc).isoformat()
        review_result = reviewer.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Review the continuation."}
                ],
                "code_task": contract().model_dump(mode="json"),
                "code_candidate": candidate(2).model_dump(mode="json"),
                "code_review_loop": published_loop.model_dump(mode="json"),
                "code_worker_submission": result["code_worker_submission"],
                "code_artifact_manifest": {
                    "manifest_id": "manifest-1",
                    "candidate": candidate(2).model_dump(mode="json"),
                    "files": [],
                    "created_at": published_at,
                },
                "code_publication_receipt": {
                    "publication_id": "publication-1",
                    "idempotency_key": "publish-once",
                    "candidate": candidate(2).model_dump(mode="json"),
                    "manifest_id": "manifest-1",
                    "target_root": "D:/target",
                    "base_revision": "sha256:base",
                    "applied_revision": "sha256:applied",
                    "applied_at": published_at,
                },
            }
        )

        self.assertEqual(review_result["code_review_loop"]["status"], "APPLIED")
        self.assertIn(
            "Retest the clarified behavior.",
            reviewer_model.last_request_text,
        )
        self.assertNotIn(
            "Update only the existing behavior.",
            reviewer_model.last_request_text,
        )

    def test_reviewer_requests_bounded_repair_and_stops(self) -> None:
        model = ToolRecordingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": REQUEST_CODE_WORKER_REPAIR_NAME,
                            "args": {
                                "summary": "The focused check failed.",
                                "required_changes": [
                                    "Fix the feature output."
                                ],
                                "findings": [
                                    {
                                        "finding_id": "wrong_output",
                                        "category": "CANDIDATE_DEFECT",
                                        "summary": "The output is incorrect.",
                                        "affected_requirement_ids": [
                                            "feature_works"
                                        ],
                                        "related_paths": ["app.py"],
                                    }
                                ],
                                "preserve_behaviors": [
                                    "Keep the public entrypoint."
                                ],
                            },
                            "id": "repair-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        reviewer = create_code_reviewer(model, tools=[code_probe])
        loop = create_code_review_loop(
            candidate=candidate(),
            worker_checkpoint_id="worker-thread-1",
            reviewer_checkpoint_id="reviewer-thread-1",
        )
        result = reviewer.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Review it."}
                ],
                "code_task": contract().model_dump(mode="json"),
                "code_candidate": candidate().model_dump(mode="json"),
                "code_review_loop": loop.model_dump(mode="json"),
                "code_worker_submission": {
                    "submission": {
                        "candidate": candidate().model_dump(mode="json"),
                        "summary": "Implemented it.",
                        "requirement_status": {
                            "feature_works": "MET"
                        },
                    }
                },
            }
        )

        self.assertTrue(result["code_agent_finished"])
        self.assertEqual(
            result["code_review_loop"]["status"],
            "WAITING_FOR_WORKER",
        )
        self.assertEqual(
            result["code_repair_instruction"]["round_no"],
            1,
        )
        self.assertIn("execute", CODE_REVIEWER_FILESYSTEM_TOOLS)
        self.assertIn(
            PUBLISH_REVIEWED_CANDIDATE_NAME,
            model.bound_tool_names,
        )
        self.assertNotIn("execute", model.bound_tool_names)
        self.assertNotIn("task", model.bound_tool_names)
        self.assertEqual(model.invocation_count, 1)


if __name__ == "__main__":
    unittest.main()
