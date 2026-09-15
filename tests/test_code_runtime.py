"""Provider- and Docker-free tests for the CODE orchestration adapter."""

from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["PHOENIX_TRACING_ENABLED"] = "false"

from langchain_core.messages import AIMessage

from eventing.run_pump import EventPauseControl, EventRunCancelled
from planning_models import CodeRequirement, CodeTaskContract
from workers.code_publisher import (
    CodePublicationContext,
    publish_code_artifacts,
)
from workers.code_review_models import (
    CodeCandidateRef,
    CodeCheckResult,
    CodeChangedFile,
    CodeContinuationSubmission,
    CodeReviewLoopState,
    CodeReviewReport,
    CodeWorkerSubmission,
    SchedulerCodeDecision,
    finish_code_review,
    receive_code_continuation_submission,
    record_code_publication,
)
from workers.code_runtime import CodeStepRuntime
from workers.code_runtime_checkpoint import CodeRuntimeCheckpointStore
from workers.docker_sandbox import CodeSandboxPair


def contract() -> CodeTaskContract:
    return CodeTaskContract(
        requirements=[
            CodeRequirement(
                requirement_id="feature",
                statement="Publish the requested implementation.",
            )
        ],
        validation_expectations=["Run one focused check."],
    )


class FakeSandboxManager:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.cleaned = False

    def create_pair(
        self,
        workspace_id: str,
        *,
        handoff_root: Path | None = None,
    ) -> CodeSandboxPair:
        self.events.append("create")
        self.handoff_root = handoff_root
        return CodeSandboxPair(
            pair_id="pair-1",
            workspace_id=workspace_id,
            candidate_volume="candidate-volume",
            review_volume="review-volume",
            worker_container="worker-container",
            reviewer_container="reviewer-container",
            image="test-image",
            active_role="WORKER",
        )

    def copy_source(self, pair, source):
        self.events.append("copy_source")
        self.source = Path(source)
        return pair

    def worker_backend(self, pair):
        return object()

    def reviewer_backend(self, pair):
        return object()

    def handoff(self, pair, role):
        self.events.append(f"handoff:{role}")
        return replace(pair, active_role=role)

    def export_candidate(self, pair, destination: Path):
        self.events.append("export_candidate")
        destination.mkdir(parents=True)
        (destination / "app.py").write_text(
            "print('reviewed candidate')\n",
            encoding="utf-8",
        )
        return destination

    def freeze(self, pair):
        self.events.append("freeze")
        return replace(pair, active_role=None)

    def recover_pair(
        self,
        pair,
        *,
        candidate_snapshot: Path | None,
        reviewer_snapshot: Path | None,
    ):
        self.events.append("recover_pair")
        self.recovered_candidate_snapshot = candidate_snapshot
        self.recovered_reviewer_snapshot = reviewer_snapshot
        return replace(pair, active_role=None), False

    def export_review(self, pair, destination: Path):
        self.events.append("export_review")
        destination.mkdir(parents=True)
        (destination / "focused-check.txt").write_text(
            "passed\n",
            encoding="utf-8",
        )
        return destination

    def cleanup(self, pair):
        self.events.append("cleanup")
        self.cleaned = True


class FakeWorkerGraph:
    async def ainvoke(self, state, config=None):
        candidate = CodeCandidateRef.model_validate(state["code_candidate"])
        loop = CodeReviewLoopState.model_validate(state["code_review_loop"])
        submission = CodeWorkerSubmission(
            candidate=candidate,
            summary="Implemented the feature.",
            requirement_status={"feature": "MET"},
            changed_files=(
                CodeChangedFile(
                    path="app.py",
                    change_summary="Implemented the requested behavior.",
                ),
            ),
            proposed_artifact_paths=("app.py",),
        )
        return {
            "messages": [AIMessage(content="Candidate submitted.")],
            "executor_model_calls_used": 1,
            "executor_tool_calls_used": 2,
            "show_all_toolsets_calls_used": 0,
            "code_candidate": candidate.model_dump(mode="json"),
            "code_review_loop": loop.model_dump(mode="json"),
            "code_worker_submission": {
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "worker_id": state["worker_id"],
                "submission": submission.model_dump(mode="json"),
                "resolved_evidence": [],
            },
        }


class CapturingWorkerGraph(FakeWorkerGraph):
    def __init__(self) -> None:
        self.seen_states: list[dict] = []

    async def ainvoke(self, state, config=None):
        self.seen_states.append(dict(state))
        return await super().ainvoke(state, config=config)


class FakeReviewerGraph:
    async def ainvoke(self, state, config=None):
        candidate = CodeCandidateRef.model_validate(state["code_candidate"])
        loop = CodeReviewLoopState.model_validate(state["code_review_loop"])
        context = CodePublicationContext.model_validate(
            state["code_publication_context"]
        )
        manifest, receipt = publish_code_artifacts(
            candidate=candidate,
            approved_artifact_paths=("app.py",),
            context=context,
        )
        loop = record_code_publication(loop, candidate=candidate)
        report = CodeReviewReport(
            candidate=candidate,
            verdict="PASSED",
            summary="The implementation was verified and published.",
            verification_summary="The focused behavior check passed.",
            verified_requirement_ids=("feature",),
            check_results=(
                CodeCheckResult(
                    check_id="focused-check",
                    description="Run the focused behavior check.",
                    status="PASSED",
                    summary="Observed the expected output.",
                ),
            ),
            changed_files=(
                CodeChangedFile(
                    path="app.py",
                    change_summary="Implemented the requested behavior.",
                ),
            ),
            approved_artifact_paths=("app.py",),
            published_artifact_paths=("app.py",),
            delivery_location=receipt.target_root,
            publication_id=receipt.publication_id,
            applied_revision=receipt.applied_revision,
        )
        loop = finish_code_review(loop, report)
        return {
            "messages": [AIMessage(content=report.summary)],
            "executor_model_calls_used": 2,
            "executor_tool_calls_used": 3,
            "show_all_toolsets_calls_used": 0,
            "code_review_loop": loop.model_dump(mode="json"),
            "code_review_report": report.model_dump(mode="json"),
            "code_artifact_manifest": manifest.model_dump(mode="json"),
            "code_publication_receipt": receipt.model_dump(mode="json"),
        }


def failed_review(
    loop: CodeReviewLoopState,
    *,
    recommended_action: str = "CONTINUE",
) -> tuple[CodeReviewLoopState, CodeReviewReport]:
    report = CodeReviewReport(
        candidate=loop.candidate,
        verdict="FAILED",
        summary="The candidate still needs a focused repair.",
        verification_summary="The focused behavior check failed.",
        check_results=(
            CodeCheckResult(
                check_id="focused-check",
                description="Run the focused behavior check.",
                status="FAILED",
                summary="Observed the old behavior.",
            ),
        ),
        failed_test_summaries=("focused-check: old behavior",),
        changed_files=(
            CodeChangedFile(
                path="app.py",
                change_summary="Candidate requires another revision.",
            ),
        ),
        recommended_action=recommended_action,
    )
    return finish_code_review(loop, report), report


class ContinueWorkerGraph(FakeWorkerGraph):
    async def ainvoke(self, state, config=None):
        loop = CodeReviewLoopState.model_validate(state["code_review_loop"])
        if loop.pending_scheduler_directive is None:
            return await super().ainvoke(state, config=config)

        old = loop.candidate
        candidate = old.model_copy(
            update={"candidate_revision": old.candidate_revision + 1}
        )
        continuation = CodeContinuationSubmission(
            scheduler_epoch=loop.scheduler_epoch,
            action="REVISION_READY",
            candidate=candidate,
            summary="Applied the Scheduler's focused repair instruction.",
            changed_files=(
                CodeChangedFile(
                    path="app.py",
                    change_summary="Repaired the focused behavior.",
                ),
            ),
        )
        loop = receive_code_continuation_submission(loop, continuation)
        submission = CodeWorkerSubmission(
            candidate=candidate,
            summary="Implemented the feature after Scheduler guidance.",
            requirement_status={"feature": "MET"},
            changed_files=continuation.changed_files,
            proposed_artifact_paths=("app.py",),
        )
        submission_record = dict(state["code_worker_submission"])
        submission_record["submission"] = submission.model_dump(mode="json")
        return {
            "messages": [AIMessage(content="Continued candidate submitted.")],
            "executor_model_calls_used": 1,
            "executor_tool_calls_used": 1,
            "show_all_toolsets_calls_used": 0,
            "code_candidate": candidate.model_dump(mode="json"),
            "code_review_loop": loop.model_dump(mode="json"),
            "code_worker_submission": submission_record,
        }


class FailOnceReviewerGraph(FakeReviewerGraph):
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, state, config=None):
        self.calls += 1
        if self.calls > 1:
            return await super().ainvoke(state, config=config)
        loop = CodeReviewLoopState.model_validate(state["code_review_loop"])
        loop, report = failed_review(loop)
        return {
            "messages": [AIMessage(content=report.summary)],
            "executor_model_calls_used": 1,
            "executor_tool_calls_used": 1,
            "show_all_toolsets_calls_used": 0,
            "code_review_loop": loop.model_dump(mode="json"),
            "code_review_report": report.model_dump(mode="json"),
        }


class CodeRuntimeTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def runtime_input() -> dict:
        return {
            "messages": [{"role": "user", "content": "Implement it."}],
            "event_id": "event-code-control",
            "step_id": "1",
            "worker_id": "worker-code-control",
            "code_task": contract().model_dump(mode="json"),
            "executor_model_run_limit": 10,
            "executor_tool_run_limit": 10,
            "show_all_toolsets_run_limit": 0,
        }

    async def test_worker_export_reviewer_publish_and_archive(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text(
                "print('base')\n",
                encoding="utf-8",
            )
            manager = FakeSandboxManager()

            independent_reviewer, independent_summary = object(), object()
            reviewer_factory_calls = []
            def independent_factory(model, **kwargs):
                self.assertIs(model, independent_reviewer)
                self.assertIs(kwargs["summary_model"], independent_summary)
                reviewer_factory_calls.append(model)
                return FakeReviewerGraph()

            runtime = CodeStepRuntime(
                object(),
                reviewer_model=independent_reviewer,
                reviewer_summary_model=independent_summary,
                sandbox_manager=manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=independent_factory,
            )
            result = await runtime.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": "Implement it."}
                    ],
                    "event_id": "event-code",
                    "step_id": "1",
                    "worker_id": "worker-code",
                    "code_task": contract().model_dump(mode="json"),
                    "executor_model_run_limit": 10,
                    "executor_tool_run_limit": 10,
                    "show_all_toolsets_run_limit": 0,
                },
                config={"configurable": {"thread_id": "code-thread"}},
            )

            self.assertEqual(
                result["code_review_loop"]["status"],
                "APPLIED",
            )
            self.assertEqual(
                result["code_review_report"]["verdict"],
                "PASSED",
            )
            self.assertTrue((manager.source / ".git").is_dir())
            self.assertIn("code_integration_commit", result)
            self.assertEqual(
                result["code_integration_status"]["head_commit"],
                result["code_integration_commit"]["accepted_commit"],
            )
            self.assertTrue(
                result["code_integration_status"]["working_tree_clean"]
            )
            self.assertEqual(
                (workspace / "app.py").read_text(encoding="utf-8"),
                "print('base')\n",
            )
            handoff_receipts = result["code_handoff_publication_receipts"]
            self.assertEqual(len(handoff_receipts), 1)
            self.assertEqual(
                Path(handoff_receipts[0]["storage_path"]).read_text(
                    encoding="utf-8"
                ),
                "print('reviewed candidate')\n",
            )
            integration_root = Path(
                result["code_publication_receipt"]["target_root"]
            )
            self.assertEqual(
                (integration_root / "app.py").read_text(encoding="utf-8"),
                "print('reviewed candidate')\n",
            )
            self.assertEqual(result["executor_model_calls_used"], 3)
            self.assertEqual(result["executor_tool_calls_used"], 5)
            self.assertIsNotNone(result["code_attempt_final_record"])
            archive_root = Path(result["code_attempt_archive"]["root_path"])
            self.assertTrue((archive_root / "final_record.json").is_file())
            runtime_checkpoint = CodeRuntimeCheckpointStore().load(archive_root)
            self.assertIsNotNone(runtime_checkpoint)
            self.assertEqual(runtime_checkpoint.phase, "TERMINAL")
            self.assertTrue(runtime_checkpoint.cleanup_authorized)
            self.assertEqual(runtime_checkpoint.sandbox.pair_id, "pair-1")
            self.assertTrue(reviewer_factory_calls)
            self.assertTrue(manager.cleaned)
            self.assertIsNotNone(manager.handoff_root)
            self.assertTrue(manager.handoff_root.is_dir())
            self.assertEqual(
                manager.events,
                [
                    "create",
                    "copy_source",
                    "handoff:REVIEWER",
                    "export_candidate",
                    "freeze",
                    "export_review",
                    "cleanup",
                ],
            )

            cleanup = runtime.cleanup_expired_archives(
                protected_run_ids=set(),
                now=datetime.now(timezone.utc) + timedelta(days=11),
            )
            self.assertEqual(cleanup[0]["status"], "CLEANED")
            self.assertFalse(
                (
                    archive_root
                    / result["code_attempt_archive"]["candidate_snapshot_path"]
                ).exists()
            )
            self.assertFalse(
                (
                    archive_root
                    / result["code_attempt_archive"]["reviewer_snapshot_path"]
                ).exists()
            )
            self.assertTrue((archive_root / "final_record.json").is_file())
            self.assertTrue((archive_root / "runtime_checkpoint.json").is_file())
            self.assertTrue((archive_root / "retention_cleanup.json").is_file())

    async def test_insert_pauses_code_between_worker_and_reviewer(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text("print('base')\n", encoding="utf-8")
            manager = FakeSandboxManager()
            runtime = CodeStepRuntime(
                object(),
                sandbox_manager=manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: FakeReviewerGraph(),
            )
            control = EventPauseControl()
            control.request()
            running = asyncio.create_task(
                runtime.ainvoke(
                    self.runtime_input(),
                    config={
                        "configurable": {
                            "thread_id": "code-insert",
                            "event_pause_control": control,
                        }
                    },
                )
            )
            await asyncio.wait_for(control.wait_until_paused(), timeout=10)
            attempt_root = next((root / "archives").iterdir())
            paused = CodeRuntimeCheckpointStore().load(attempt_root)
            self.assertEqual(paused.phase, "CANDIDATE_READY")
            self.assertIsNone(paused.sandbox.active_role)
            self.assertIsNotNone(paused.recovery_state)
            self.assertIn("freeze", manager.events)

            control.resume()
            result = await asyncio.wait_for(running, timeout=30)
            self.assertEqual(result["code_review_loop"]["status"], "APPLIED")
            self.assertTrue(manager.cleaned)

    async def test_restart_resumes_paused_candidate_at_reviewer(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text("print('base')\n", encoding="utf-8")
            first_manager = FakeSandboxManager()
            first_runtime = CodeStepRuntime(
                object(),
                sandbox_manager=first_manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: FakeReviewerGraph(),
            )
            control = EventPauseControl()
            control.request()
            interrupted = asyncio.create_task(
                first_runtime.ainvoke(
                    self.runtime_input(),
                    config={
                        "configurable": {
                            "thread_id": "code-insert-restart",
                            "event_pause_control": control,
                        }
                    },
                )
            )
            await asyncio.wait_for(control.wait_until_paused(), timeout=10)
            interrupted.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await interrupted

            interrupted_checkpoint = CodeRuntimeCheckpointStore().load(
                next((root / "archives").iterdir())
            )
            self.assertEqual(interrupted_checkpoint.phase, "RECOVERY_REQUIRED")
            self.assertIsNotNone(interrupted_checkpoint.recovery_state)

            recovered_manager = FakeSandboxManager()
            independent_reviewer, independent_summary = object(), object()
            reviewer_factory_calls = []
            def independent_factory(model, **kwargs):
                self.assertIs(model, independent_reviewer)
                self.assertIs(kwargs["summary_model"], independent_summary)
                reviewer_factory_calls.append(model)
                return FakeReviewerGraph()

            recovered_runtime = CodeStepRuntime(
                object(),
                reviewer_model=independent_reviewer,
                reviewer_summary_model=independent_summary,
                sandbox_manager=recovered_manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=independent_factory,
            )
            outcomes = recovered_runtime.recover_startup_sessions()
            self.assertEqual(outcomes[0]["status"], "RECOVERED")

            result = await recovered_runtime.ainvoke(
                self.runtime_input(),
                config={
                    "configurable": {"thread_id": "code-insert-restart"}
                },
            )
            self.assertEqual(result["code_review_loop"]["status"], "APPLIED")
            self.assertEqual(recovered_manager.events.count("recover_pair"), 1)
            self.assertNotIn("create", recovered_manager.events)
            self.assertTrue(reviewer_factory_calls)
            self.assertTrue(recovered_manager.cleaned)

    async def test_cancel_reconciles_recovery_checkpoint_before_cleanup(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text("print('base')\n", encoding="utf-8")
            first_runtime = CodeStepRuntime(
                object(),
                sandbox_manager=FakeSandboxManager(),
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: FakeReviewerGraph(),
            )
            control = EventPauseControl()
            control.request_cancel("cancel-event")
            interrupted = asyncio.create_task(
                first_runtime.ainvoke(
                    self.runtime_input(),
                    config={
                        "configurable": {
                            "thread_id": "code-cancel-reconcile",
                            "event_pause_control": control,
                        }
                    },
                )
            )
            await asyncio.wait_for(control.wait_until_paused(), timeout=10)
            control.terminate()
            with self.assertRaises(EventRunCancelled):
                await interrupted

            attempt_root = next((root / "archives").iterdir())
            interrupted_checkpoint = CodeRuntimeCheckpointStore().load(
                attempt_root
            )
            self.assertEqual(interrupted_checkpoint.phase, "RECOVERY_REQUIRED")

            recovered_manager = FakeSandboxManager()
            recovered_runtime = CodeStepRuntime(
                object(),
                sandbox_manager=recovered_manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: FakeReviewerGraph(),
            )
            outcomes = recovered_runtime.recover_startup_sessions()
            self.assertEqual(outcomes[0]["status"], "RECOVERED")

            records = recovered_runtime.cancel_run("event-code-control")

            self.assertEqual(records[0]["outcome"], "CANCELLED")
            terminal = CodeRuntimeCheckpointStore().load(attempt_root)
            self.assertEqual(terminal.phase, "TERMINAL")
            self.assertTrue(terminal.cleanup_authorized)
            self.assertEqual(recovered_manager.events.count("recover_pair"), 1)
            self.assertTrue(recovered_manager.cleaned)

    async def test_second_code_step_starts_from_first_accepted_commit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text(
                "print('base')\n",
                encoding="utf-8",
            )
            manager = FakeSandboxManager()
            worker = CapturingWorkerGraph()
            runtime = CodeStepRuntime(
                object(),
                sandbox_manager=manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: worker,
                reviewer_factory=lambda *args, **kwargs: FakeReviewerGraph(),
            )

            def input_state(step_id: int) -> dict:
                return {
                    "messages": [
                        {"role": "user", "content": f"Implement step {step_id}."}
                    ],
                    "event_id": "event-code-chain",
                    "step_id": str(step_id),
                    "worker_id": f"worker-code-{step_id}",
                    "code_task": contract().model_dump(mode="json"),
                    "executor_model_run_limit": 10,
                    "executor_tool_run_limit": 10,
                    "show_all_toolsets_run_limit": 0,
                }

            first = await runtime.ainvoke(
                input_state(1),
                config={"configurable": {"thread_id": "code-thread-1"}},
            )
            second = await runtime.ainvoke(
                input_state(2),
                config={"configurable": {"thread_id": "code-thread-2"}},
            )

            first_commit = first["code_integration_commit"]["accepted_commit"]
            self.assertEqual(
                worker.seen_states[1]["code_integration_status"]["head_commit"],
                first_commit,
            )
            self.assertEqual(
                worker.seen_states[1]["code_integration_status"][
                    "accepted_commit_count"
                ],
                1,
            )
            self.assertEqual(
                second["code_integration_commit"]["parent_commit"],
                first_commit,
            )
            self.assertNotEqual(
                second["code_integration_commit"]["accepted_commit"],
                first_commit,
            )

    async def test_continue_reuses_frozen_pair_and_applies_revision(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text("print('base')\n", encoding="utf-8")
            manager = FakeSandboxManager()
            reviewer = FailOnceReviewerGraph()
            runtime = CodeStepRuntime(
                object(),
                sandbox_manager=manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: ContinueWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: reviewer,
            )
            first = await runtime.ainvoke(
                self.runtime_input(),
                config={"configurable": {"thread_id": "code-continue"}},
            )
            self.assertEqual(
                first["code_review_loop"]["status"],
                "ESCALATED_TO_SCHEDULER",
            )
            self.assertIsNotNone(first["code_runtime_session_id"])
            self.assertFalse(manager.cleaned)
            attempt_roots = list((root / "archives").iterdir())
            self.assertEqual(len(attempt_roots), 1)
            paused_checkpoint = CodeRuntimeCheckpointStore().load(
                attempt_roots[0]
            )
            self.assertIsNotNone(paused_checkpoint)
            self.assertEqual(paused_checkpoint.phase, "AWAITING_SCHEDULER")
            self.assertFalse(paused_checkpoint.cleanup_authorized)
            self.assertIsNone(paused_checkpoint.sandbox.active_role)

            continued = {
                **self.runtime_input(),
                "code_runtime_session_id": first["code_runtime_session_id"],
                "code_scheduler_decision": SchedulerCodeDecision(
                    action="CONTINUE",
                    reason="The defect is local and the same context is valuable.",
                    worker_instruction="Repair the focused behavior only.",
                    reviewer_instruction="Re-run the focused check.",
                    repair_rounds=1,
                ).model_dump(mode="json"),
            }
            second = await runtime.ainvoke(
                continued,
                config={"configurable": {"thread_id": "code-continue"}},
            )

            self.assertEqual(second["code_review_loop"]["status"], "APPLIED")
            self.assertEqual(
                second["code_review_loop"]["candidate"]["candidate_revision"],
                2,
            )
            self.assertEqual(
                len(second["code_review_loop"]["scheduler_continuations"]),
                1,
            )
            self.assertEqual(
                second["code_scheduler_decision_applied"]["action"],
                "CONTINUE",
            )
            self.assertEqual(manager.events.count("create"), 1)
            self.assertTrue(manager.cleaned)

    async def test_startup_rebuilds_in_memory_session_then_continues(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "app.py").write_text("print('base')\n", encoding="utf-8")
            first_manager = FakeSandboxManager()
            first_runtime = CodeStepRuntime(
                object(),
                sandbox_manager=first_manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: ContinueWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: FailOnceReviewerGraph(),
            )
            first = await first_runtime.ainvoke(
                self.runtime_input(),
                config={"configurable": {"thread_id": "code-recover"}},
            )
            session_id = first["code_runtime_session_id"]

            recovered_manager = FakeSandboxManager()
            recovered_runtime = CodeStepRuntime(
                object(),
                sandbox_manager=recovered_manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: ContinueWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: FakeReviewerGraph(),
            )
            outcomes = recovered_runtime.recover_startup_sessions()

            self.assertEqual(
                outcomes,
                (
                    {
                        "status": "RECOVERED",
                        "session_id": session_id,
                        "resource_action": "REUSED",
                    },
                ),
            )
            self.assertEqual(recovered_runtime.recovered_session_ids, (session_id,))
            self.assertEqual(recovered_manager.events, ["recover_pair"])
            self.assertTrue(
                recovered_manager.recovered_candidate_snapshot.is_dir()
            )

            continued = await recovered_runtime.ainvoke(
                {
                    **self.runtime_input(),
                    "code_runtime_session_id": session_id,
                    "code_scheduler_decision": SchedulerCodeDecision(
                        action="CONTINUE",
                        reason="Resume the safely frozen attempt.",
                        worker_instruction="Repair the focused behavior only.",
                        reviewer_instruction="Re-run the focused check.",
                        repair_rounds=1,
                    ).model_dump(mode="json"),
                },
                config={"configurable": {"thread_id": "code-recover"}},
            )
            self.assertEqual(continued["code_review_loop"]["status"], "APPLIED")
            self.assertTrue(recovered_manager.cleaned)

    async def test_stop_archives_and_releases_frozen_pair(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = FakeSandboxManager()
            reviewer = FailOnceReviewerGraph()
            runtime = CodeStepRuntime(
                object(),
                sandbox_manager=manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: reviewer,
            )
            first = await runtime.ainvoke(
                self.runtime_input(),
                config={"configurable": {"thread_id": "code-stop"}},
            )
            stopped = await runtime.ainvoke(
                {
                    **self.runtime_input(),
                    "code_runtime_session_id": first["code_runtime_session_id"],
                    "code_scheduler_decision": SchedulerCodeDecision(
                        action="STOP",
                        reason="The remaining work is outside the accepted scope.",
                    ).model_dump(mode="json"),
                },
                config={"configurable": {"thread_id": "code-stop"}},
            )
            self.assertEqual(stopped["code_review_loop"]["status"], "STOPPED")
            self.assertEqual(
                stopped["code_attempt_final_record"]["outcome"],
                "FAILED",
            )
            self.assertEqual(
                stopped["code_attempt_final_record"]["scheduler_decision"][
                    "action"
                ],
                "STOP",
            )
            self.assertIsNotNone(stopped["code_attempt_archive"])
            self.assertTrue(manager.cleaned)

    async def test_event_cancel_archives_and_releases_frozen_pair(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = FakeSandboxManager()
            runtime = CodeStepRuntime(
                object(),
                sandbox_manager=manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: FailOnceReviewerGraph(),
            )
            first = await runtime.ainvoke(
                self.runtime_input(),
                config={"configurable": {"thread_id": "code-event-cancel"}},
            )
            self.assertIsNotNone(first["code_runtime_session_id"])

            records = runtime.cancel_run(
                "event-code-control",
                reason="The user cancelled the owning Event.",
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["outcome"], "CANCELLED")
            self.assertIsNone(records[0]["scheduler_decision"])
            archive_root = Path(records[0]["archive"]["root_path"])
            self.assertTrue((archive_root / "final_record.json").is_file())
            checkpoint = CodeRuntimeCheckpointStore().load(archive_root)
            self.assertEqual(checkpoint.phase, "TERMINAL")
            self.assertTrue(checkpoint.cleanup_authorized)
            self.assertTrue(manager.cleaned)
            self.assertNotIn(
                first["code_runtime_session_id"],
                runtime.recovered_session_ids,
            )

    async def test_event_replace_archives_pair_as_superseded(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = FakeSandboxManager()
            runtime = CodeStepRuntime(
                object(),
                sandbox_manager=manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: FailOnceReviewerGraph(),
            )
            first = await runtime.ainvoke(
                self.runtime_input(),
                config={"configurable": {"thread_id": "code-event-replace"}},
            )

            records = runtime.supersede_run(
                "event-code-control",
                reason="A newer user request replaced this Event.",
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["outcome"], "SUPERSEDED")
            self.assertIsNone(records[0]["scheduler_decision"])
            self.assertTrue(manager.cleaned)
            self.assertNotIn(
                first["code_runtime_session_id"],
                runtime.recovered_session_ids,
            )

    async def test_restart_supersedes_old_attempt_and_starts_fresh_pair(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = FakeSandboxManager()
            reviewer = FailOnceReviewerGraph()
            runtime = CodeStepRuntime(
                object(),
                sandbox_manager=manager,
                source_root=workspace,
                target_root=workspace,
                archive_root=root / "archives",
                run_storage_root=root / "runs",
                worker_factory=lambda *args, **kwargs: FakeWorkerGraph(),
                reviewer_factory=lambda *args, **kwargs: reviewer,
            )
            first = await runtime.ainvoke(
                self.runtime_input(),
                config={"configurable": {"thread_id": "code-restart-old"}},
            )
            restarted = await runtime.ainvoke(
                {
                    **self.runtime_input(),
                    "code_runtime_session_id": first["code_runtime_session_id"],
                    "code_scheduler_decision": SchedulerCodeDecision(
                        action="RESTART",
                        reason="The implementation direction is structurally wrong.",
                    ).model_dump(mode="json"),
                },
                config={"configurable": {"thread_id": "code-restart-new"}},
            )

            self.assertEqual(restarted["code_review_loop"]["status"], "APPLIED")
            self.assertEqual(manager.events.count("create"), 2)
            superseded = restarted["code_superseded_attempt_records"]
            self.assertEqual(len(superseded), 1)
            self.assertEqual(superseded[0]["outcome"], "CANCELLED")
            self.assertEqual(
                superseded[0]["scheduler_decision"]["action"],
                "RESTART",
            )
            self.assertEqual(
                restarted["code_attempt_final_record"]["parent_attempt_id"],
                superseded[0]["candidate"]["attempt_id"],
            )
            self.assertTrue(manager.cleaned)


if __name__ == "__main__":
    unittest.main()
