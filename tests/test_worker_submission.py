"""Provider-free tests for Worker review submission and evidence resolution."""

from __future__ import annotations

import unittest
import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field
from deepagents.backends.utils import create_file_data

from prompt_loader import load_prompt
from workers.web_worker import create_web_worker
from artifact_models import (
    DownloadedArtifactRecord,
    WorkerArtifactCandidate,
)
from middlewares import DynamicExecutionBudgetMiddleware
from workers.submission import (
    SUBMIT_FOR_REVIEW_NAME,
    resolve_artifact_candidates,
    resolve_tool_evidence,
)


@tool
def web_search(value: str) -> str:
    """Return stable evidence for a Worker submission test."""

    return f"observed:{value}"


class SubmissionScriptModel(BaseChatModel):
    """Call one business tool, then submit that exact result for review."""

    invocation_count: int = 0
    bound_tool_names: list[str] = Field(default_factory=list)
    bound_tool_history: list[list[str]] = Field(default_factory=list)

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda
        assert schema.__name__ == "SkillChoice"
        return RunnableLambda(lambda _: {"parsed": schema.model_validate({"skill_ids": [], "reason": "Offline fixture"})})

    @property
    def _llm_type(self) -> str:
        return "worker-submission-script-test"

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [current_tool.name for current_tool in tools]
        self.bound_tool_history.append(list(self.bound_tool_names))
        return self

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        invocation_index = self.invocation_count
        self.invocation_count += 1

        if invocation_index == 0:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"value": "cinema-source"},
                        "id": "evidence-call-1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": SUBMIT_FOR_REVIEW_NAME,
                        "args": {
                            "submission": {
                                "summary": "Found the requested source.",
                                "final_conclusion": (
                                    "The cited tool result supports the Step."
                                ),
                                "criterion_claims": [
                                    {
                                        "criterion": "Find one source.",
                                        "conclusion": "One source was found.",
                                        "evidence_tool_call_ids": [
                                            "evidence-call-1"
                                        ],
                                    }
                                ],
                                "artifact_candidates": [],
                                "unresolved_items": [],
                            }
                        },
                        "id": "submission-call-1",
                        "type": "tool_call",
                    }
                ],
            )

        return ChatResult(
            generations=[ChatGeneration(message=message)]
        )


class WorkerSubmissionTests(unittest.TestCase):
    def test_nested_worker_prompt_is_loaded(self) -> None:
        prompt = load_prompt("workers/web_worker")
        self.assertIn("submit_for_review", prompt)

        with self.assertRaises(ValueError):
            load_prompt("../secret")

    def test_resolver_requires_real_request_and_result_pair(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "cinema"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="{\"results\":[{\"title\":\"Cinema\"}]}",
                tool_call_id="call-1",
                name="web_search",
            ),
        ]

        evidence = resolve_tool_evidence(messages, ["call-1"])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].tool_name, "web_search")
        self.assertIn("Cinema", evidence[0].result)

        with self.assertRaisesRegex(ValueError, "unknown Tool Call ID"):
            resolve_tool_evidence(messages, ["missing-call"])

    def test_control_tools_cannot_be_cited(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "publish_worker_progress",
                        "args": {},
                        "id": "control-call",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="checkpoint",
                tool_call_id="control-call",
                name="publish_worker_progress",
            ),
        ]

        with self.assertRaisesRegex(ValueError, "control tool"):
            resolve_tool_evidence(messages, ["control-call"])

    def test_worker_submission_stops_at_safe_point(self) -> None:
        model = SubmissionScriptModel()
        worker = create_web_worker(
            model,
            tools=[web_search],
        )

        result = worker.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Find one source and submit it.",
                    }
                ],
                "worker_id": "worker-submit-1",
                "event_id": "event-submit-1",
                "step_id": "1",
            }
        )

        self.assertIn(SUBMIT_FOR_REVIEW_NAME, model.bound_tool_names)
        self.assertTrue(result["worker_review_requested"])
        record = result["worker_submission"]
        self.assertEqual(record["worker_id"], "worker-submit-1")
        self.assertEqual(
            record["submission"]["criterion_claims"][0][
                "evidence_tool_call_ids"
            ],
            ["evidence-call-1"],
        )
        self.assertEqual(
            record["resolved_evidence"][0]["tool_call_id"],
            "evidence-call-1",
        )
        self.assertEqual(model.invocation_count, 2)

    def test_model_budget_reserves_one_submit_only_round(self) -> None:
        model = SubmissionScriptModel()
        worker = create_web_worker(
            model,
            tools=[web_search],
            middleware=[
                DynamicExecutionBudgetMiddleware(
                    enable_worker_finalization=True,
                    finalization_model_rounds=1,
                )
            ],
        )

        result = worker.invoke(
            {
                "messages": [{"role": "user", "content": "Use one tool."}],
                "worker_id": "worker-budget-finalize",
                "event_id": "event-budget-finalize",
                "step_id": "1",
                "executor_model_run_limit": 1,
                "executor_tool_run_limit": 1,
                "show_all_toolsets_run_limit": 0,
            }
        )

        self.assertTrue(result["worker_review_requested"])
        self.assertEqual(result["executor_model_calls_used"], 1)
        self.assertEqual(result["worker_finalization_model_calls_used"], 1)
        self.assertEqual(
            model.bound_tool_history[-1],
            [SUBMIT_FOR_REVIEW_NAME],
        )

    def test_replace_state_gets_one_submit_only_round(self) -> None:
        model = SubmissionScriptModel(invocation_count=1)
        worker = create_web_worker(
            model,
            tools=[web_search],
        )

        result = worker.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "This attempt has been replaced; finalize it.",
                    },
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "web_search",
                                "args": {"value": "cinema-source"},
                                "id": "evidence-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content="observed:cinema-source",
                        tool_call_id="evidence-call-1",
                        name="web_search",
                    ),
                ],
                "worker_id": "worker-replaced",
                "event_id": "event-replaced",
                "step_id": "1",
                "worker_terminal_action": "REPLACE",
                "worker_finalize_requested": True,
                "worker_finalize_reason": "LEADERSHIP_REPLACE",
            }
        )

        self.assertTrue(result["worker_review_requested"])
        self.assertEqual(result["worker_finalize_reason"], "LEADERSHIP_REPLACE")
        self.assertEqual(result["worker_finalization_model_calls_used"], 1)
        self.assertEqual(model.bound_tool_history[-1], [SUBMIT_FOR_REVIEW_NAME])

    def test_workspace_artifact_is_resolved_from_checkpoint_bytes(self) -> None:
        candidate = WorkerArtifactCandidate(
            candidate_id="workspace-note",
            output_id="research_notes",
            kind="WORKSPACE_FILE",
            description="Research handoff note",
            path="notes/research.md",
        )
        with tempfile.TemporaryDirectory() as directory:
            resolved = resolve_artifact_candidates(
                {
                    "files": {"/notes/research.md": create_file_data("hello")},
                    "event_id": "event-workspace-note",
                    "worker_id": "worker-workspace-note",
                    "run_storage_root": directory,
                },
                [candidate],
            )[0]

            self.assertEqual(Path(resolved.storage_path).read_bytes(), b"hello")

        self.assertEqual(resolved.location, "/notes/research.md")
        self.assertEqual(resolved.output_id, "research_notes")
        self.assertEqual(resolved.size_bytes, 5)
        self.assertEqual(
            resolved.sha256,
            hashlib.sha256(b"hello").hexdigest(),
        )

    def test_download_candidate_cannot_invent_runtime_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown downloaded"):
            resolve_artifact_candidates(
                {"worker_downloaded_artifacts": []},
                [
                    WorkerArtifactCandidate(
                        candidate_id="download-invented",
                        kind="DOWNLOADED_FILE",
                        description="Invented download",
                        evidence_tool_call_ids=["download-call"],
                    )
                ],
            )

    def test_download_and_remote_repository_candidates_are_typed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = b"downloaded"
            stored = Path(directory) / "source.py"
            stored.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            record = DownloadedArtifactRecord(
                candidate_id="download-real",
                source_url="https://example.com/source.py",
                storage_path=str(stored),
                filename="source.py",
                size_bytes=len(payload),
                sha256=digest,
                media_type="text/x-python",
                tool_call_id="download-call",
                downloaded_at=datetime.now(timezone.utc),
            )
            candidates = [
                WorkerArtifactCandidate(
                    candidate_id="download-real",
                    kind="DOWNLOADED_FILE",
                    description="Downloaded source",
                    evidence_tool_call_ids=["download-call"],
                ),
                WorkerArtifactCandidate(
                    candidate_id="repo-deep-agents",
                    kind="REMOTE_REPOSITORY",
                    description="Repository for later shallow clone",
                    repository_url="https://github.com/langchain-ai/deepagents",
                    repository_ref="main",
                    relevant_paths=["libs/deepagents"],
                    evidence_tool_call_ids=["search-call"],
                ),
            ]
            resolved = resolve_artifact_candidates(
                {"worker_downloaded_artifacts": [record.model_dump(mode="json")]},
                candidates,
            )

        self.assertEqual(resolved[0].sha256, digest)
        self.assertEqual(
            resolved[1].location,
            "https://github.com/langchain-ai/deepagents@main",
        )


if __name__ == "__main__":
    unittest.main()
