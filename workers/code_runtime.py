"""Closed Worker -> Reviewer -> Publisher runtime for one CODE PlanStep.

This module is the control-plane adapter between the outer Planning Graph and
the two dedicated Deep Agents.  It owns Docker lifecycle and durable exports;
it does not make implementation or review decisions itself.
"""

from __future__ import annotations

def _review_tool_audit(trace):
    from reporting.context import _tool_audit
    return _tool_audit(trace)
from runtime_tracing import operation

import asyncio
import json
import logging
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage

from agent import ask_worker
from planning_models import CodeTaskContract
from prompt_loader import render_prompt
from workers.code_attempt_models import (
    CodeArtifactManifest,
    CodeAttemptArchive,
    CodeAttemptFinalRecord,
    CodePublicationReceipt,
)
from workers.code_publisher import (
    CodePublicationContext,
    compute_code_tree_revision,
)
from workers.code_review_models import (
    CodeCandidateRef,
    CodeReviewLoopState,
    CodeReviewReport,
    SchedulerCodeDecision,
    apply_scheduler_code_decision,
    create_code_review_loop,
)
from workers.code_reviewer import create_code_reviewer
from workers.code_runtime_checkpoint import (
    CodeRuntimeCheckpoint,
    CodeRuntimeCheckpointStore,
    CodeRuntimeRecoveryState,
    CodeRuntimePhase,
    CodeSandboxPairRecord,
)
from workers.code_worker import create_code_worker
from workers.docker_sandbox import CodeSandboxManager, CodeSandboxPair
from artifact_models import ResolvedArtifactCandidate
from artifact_publisher import publish_artifact_to_handoff
from config import WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS
from integration_repository import IntegrationRepository
from run_workspace import (
    RUN_WORKSPACE_ROOT,
    initialize_code_integration,
    initialize_run_workspace,
)


AgentFactory = Callable[..., Any]
logger = logging.getLogger("agent")


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


@dataclass
class _InvocationUsage:
    """Budget and usage for one outer Planning Graph invocation."""

    model_limit: int
    tool_limit: int
    model_calls: int = 0
    tool_calls: int = 0
    show_all_toolsets_calls: int = 0

    def add(self, details: dict[str, Any]) -> None:
        summary = details.get("execution_summary", {})
        self.model_calls += max(int(summary.get("model_call_count", 0) or 0), 0)
        self.tool_calls += max(int(summary.get("tool_call_count", 0) or 0), 0)
        self.show_all_toolsets_calls += max(
            int(summary.get("show_all_toolsets_call_count", 0) or 0),
            0,
        )

    def state(
        self,
        value: dict[str, Any],
        *,
        reserve_model: int = 0,
        reserve_tools: int = 0,
    ) -> dict[str, Any]:
        available_model = self.model_limit - self.model_calls - reserve_model
        available_tools = self.tool_limit - self.tool_calls - reserve_tools
        if available_model < 1:
            raise RuntimeError(
                "CODE Step model budget cannot reach the next role while "
                "preserving its review reserve"
            )
        if available_tools < 1:
            raise RuntimeError(
                "CODE Step tool budget cannot reach the next role while "
                "preserving its review reserve"
            )
        show_all_limit = max(
            int(value.get("show_all_toolsets_run_limit", 0) or 0),
            0,
        )
        return {
            **value,
            "executor_model_run_limit": available_model,
            "executor_tool_run_limit": available_tools,
            "show_all_toolsets_run_limit": max(
                0, show_all_limit - self.show_all_toolsets_calls
            ),
        }


@dataclass
class _CodeRuntimeSession:
    """Live Worker/Reviewer pair retained while Scheduler decides recovery."""

    session_id: str
    run_id: str
    run_layout: Any
    contract: CodeTaskContract
    candidate: CodeCandidateRef
    loop: CodeReviewLoopState
    pair: CodeSandboxPair
    worker: Any
    reviewer: Any
    worker_thread_id: str
    reviewer_thread_id: str
    integration_root: Path
    integration_repository: IntegrationRepository
    base_revision: str
    archive_id: str
    attempt_root: Path
    started_at: datetime
    parent_attempt_id: str | None
    worker_id: str
    worker_submission: dict[str, Any]
    latest_candidate_relative: str = ""
    last_review_details: dict[str, Any] | None = None
    last_report: CodeReviewReport | None = None
    all_messages: list[Any] = field(default_factory=list)
    superseded_attempt_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    runtime_generation: int = 1
    latest_reviewer_checkpoint_relative: str | None = None
    recovery_phase: CodeRuntimePhase = "AWAITING_SCHEDULER"


class CodeStepRuntime:
    """Run one CODE Step through a serial, independently reviewed sandbox pair.

    The object intentionally exposes ``ainvoke`` so it fits the existing
    WorkerAgentRegistry boundary. One semaphore protects the run-local
    integration writer; Reviewer and Worker are still serial inside each pair.
    """

    def __init__(
        self,
        model,
        *,
        summary_model=None,
        reviewer_model=None,
        reviewer_summary_model=None,
        sandbox_manager: CodeSandboxManager,
        source_root: Path,
        target_root: Path,
        archive_root: Path,
        tools: Sequence[Any] = (),
        reviewer_tools: Sequence[Any] | None = None,
        max_writers: int = 1,
        max_repair_rounds: int = 2,
        archive_retention_minutes: int = 10 * 24 * 60,
        progress_every_tool_calls: int = 4,
        schema_repair_max_rounds: int = 3,
        middleware_factory: Callable[[], Sequence[Any]] | None = None,
        checkpointer=None,
        store=None,
        run_storage_root: Path = RUN_WORKSPACE_ROOT,
        worker_factory: AgentFactory = create_code_worker,
        reviewer_factory: AgentFactory = create_code_reviewer,
        runtime_checkpoint_store: CodeRuntimeCheckpointStore | None = None,
    ) -> None:
        if max_writers < 1:
            raise ValueError("max_writers must be positive")
        if max_repair_rounds < 1 or max_repair_rounds > 3:
            raise ValueError("max_repair_rounds must be between 1 and 3")
        if archive_retention_minutes < 1:
            raise ValueError("archive_retention_minutes must be positive")

        self.model = model
        self.summary_model = summary_model
        self.reviewer_model = reviewer_model if reviewer_model is not None else model
        self.reviewer_summary_model = reviewer_summary_model if reviewer_summary_model is not None else summary_model
        self.sandbox_manager = sandbox_manager
        self.source_root = source_root.resolve()
        # Reserved for the later Final Promotion boundary. Individual CODE
        # Steps publish only into the run-local integration tree.
        self.target_root = target_root.resolve()
        self.archive_root = archive_root.resolve()
        self.tools = tuple(tools)
        self.reviewer_tools = tuple(tools if reviewer_tools is None else reviewer_tools)
        self.max_repair_rounds = max_repair_rounds
        self.archive_retention_minutes = archive_retention_minutes
        self.progress_every_tool_calls = progress_every_tool_calls
        self.schema_repair_max_rounds = schema_repair_max_rounds
        self.middleware_factory = middleware_factory
        self.checkpointer = checkpointer
        self.store = store
        self.run_storage_root = Path(run_storage_root)
        self.worker_factory = worker_factory
        self.reviewer_factory = reviewer_factory
        self.runtime_checkpoint_store = (
            runtime_checkpoint_store or CodeRuntimeCheckpointStore()
        )
        self._writer_slots = asyncio.Semaphore(max_writers)
        # An escalated pair is frozen, not destroyed, until Scheduler applies
        # exactly one CONTINUE / RESTART / STOP decision.
        self._sessions: dict[str, _CodeRuntimeSession] = {}
        self._startup_recovery_blockers: list[str] = []

    @property
    def recovered_session_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sessions))

    @property
    def startup_recovery_blockers(self) -> tuple[str, ...]:
        return tuple(self._startup_recovery_blockers)

    def recover_startup_sessions(self) -> tuple[dict[str, str], ...]:
        """Rebuild safe non-terminal CODE sessions before accepting work.

        LangGraph owns the Worker/Reviewer message checkpoints. This scan
        restores the separate resource plane: run-local Git integration,
        Docker volumes/containers, and the small orchestration state needed to
        route the next Scheduler decision back to the same pair.
        """

        self.archive_root.mkdir(parents=True, exist_ok=True)
        outcomes: list[dict[str, str]] = []
        self._startup_recovery_blockers.clear()
        checkpoints: list[CodeRuntimeCheckpoint] = []
        for attempt_root in sorted(self.archive_root.iterdir()):
            if not attempt_root.is_dir():
                continue
            try:
                checkpoint = self.runtime_checkpoint_store.load(attempt_root)
            except Exception as error:
                message = (
                    f"{attempt_root.name}: unreadable runtime checkpoint "
                    f"({type(error).__name__})"
                )
                self._startup_recovery_blockers.append(message)
                outcomes.append({"status": "BLOCKED", "detail": message})
                continue
            if checkpoint is not None and checkpoint.phase != "TERMINAL":
                checkpoints.append(checkpoint)

        if len(checkpoints) > 1:
            message = (
                "Multiple non-terminal CODE attempts were found; single-writer "
                "recovery will not guess which attempt owns the integration tree"
            )
            self._startup_recovery_blockers.append(message)
            outcomes.append({"status": "BLOCKED", "detail": message})
            return tuple(outcomes)

        for checkpoint in checkpoints:
            try:
                session, rebuilt = self._restore_checkpoint_session(checkpoint)
            except Exception as error:
                message = (
                    f"{Path(checkpoint.attempt_root).name}: "
                    f"{type(error).__name__}: {error}"
                )
                self._startup_recovery_blockers.append(message)
                outcomes.append({"status": "BLOCKED", "detail": message})
                logger.exception(
                    "CODE启动恢复被安全阻止 | session_id=%s",
                    checkpoint.runtime_session_id,
                )
                continue
            self._sessions[session.session_id] = session
            outcomes.append(
                {
                    "status": "RECOVERED",
                    "session_id": session.session_id,
                    "resource_action": "REBUILT" if rebuilt else "REUSED",
                }
            )
            logger.info(
                "CODE启动恢复完成 | session_id=%s | resources=%s",
                session.session_id,
                "rebuilt" if rebuilt else "reused",
            )
        return tuple(outcomes)

    def cleanup_expired_archives(
        self,
        *,
        protected_run_ids: set[str],
        now: datetime | None = None,
    ) -> tuple[dict[str, str], ...]:
        """Prune expired Docker exports while retaining compact final records."""

        timestamp = now or datetime.now(timezone.utc)
        outcomes: list[dict[str, str]] = []
        if not self.archive_root.is_dir():
            return ()
        for attempt_root in sorted(self.archive_root.iterdir()):
            if not attempt_root.is_dir():
                continue
            final_path = attempt_root / "final_record.json"
            try:
                checkpoint = self.runtime_checkpoint_store.load(attempt_root)
                if (
                    checkpoint is None
                    or checkpoint.phase != "TERMINAL"
                    or not checkpoint.cleanup_authorized
                    or checkpoint.run_id in protected_run_ids
                    or not final_path.is_file()
                ):
                    continue
                final_record = CodeAttemptFinalRecord.model_validate_json(
                    final_path.read_text(encoding="utf-8")
                )
                archive = final_record.archive
                if archive is None or timestamp < archive.retain_until:
                    continue
                removed: list[str] = []
                for relative in (
                    archive.candidate_snapshot_path,
                    archive.reviewer_snapshot_path,
                ):
                    target = (attempt_root / relative).resolve()
                    if not target.is_relative_to(attempt_root.resolve()):
                        raise RuntimeError("archive cleanup target escaped attempt root")
                    if target.is_dir():
                        shutil.rmtree(target)
                        removed.append(relative)
                    elif target.is_file():
                        target.unlink()
                        removed.append(relative)
                cleanup_path = attempt_root / "retention_cleanup.json"
                if not cleanup_path.is_file():
                    temporary = attempt_root / f".retention-cleanup.{uuid4().hex}.tmp"
                    try:
                        temporary.write_text(
                            json.dumps(
                                {
                                    "run_id": checkpoint.run_id,
                                    "archive_id": archive.archive_id,
                                    "cleaned_at": timestamp.isoformat(),
                                    "removed": removed,
                                    "retained": [
                                        "final_record.json",
                                        self.runtime_checkpoint_store.filename,
                                    ],
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        os.replace(temporary, cleanup_path)
                    finally:
                        temporary.unlink(missing_ok=True)
                outcomes.append(
                    {
                        "status": "CLEANED",
                        "archive_id": archive.archive_id,
                        "run_id": checkpoint.run_id,
                    }
                )
            except Exception as error:
                outcomes.append(
                    {
                        "status": "SKIPPED",
                        "archive_id": attempt_root.name,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
        return tuple(outcomes)

    def cancel_run(
        self,
        run_id: str,
        *,
        reason: str = "The user cancelled the owning task.",
    ) -> tuple[dict[str, Any], ...]:
        """Archive and release every recoverable CODE pair owned by one run.

        The Event layer calls this only after the Agent stack has unwound at a
        safe point.  A just-interrupted in-flight pair therefore first exists
        as a RECOVERY_REQUIRED checkpoint; this method reconstructs that exact
        frozen pair, exports both workspaces, commits a CANCELLED final record,
        and only then authorizes Docker cleanup.
        """

        return self._terminate_run(
            run_id,
            reason=reason,
            outcome="CANCELLED",
        )

    def supersede_run(
        self,
        run_id: str,
        *,
        reason: str = "A newer user Event superseded the owning task.",
    ) -> tuple[dict[str, Any], ...]:
        """Archive old CODE resources without labeling replacement as failure."""

        return self._terminate_run(
            run_id,
            reason=reason,
            outcome="SUPERSEDED",
        )

    def _terminate_run(
        self,
        run_id: str,
        *,
        reason: str,
        outcome: str,
    ) -> tuple[dict[str, Any], ...]:
        normalized_run_id = str(run_id).strip()
        normalized_reason = str(reason).strip()
        if not normalized_run_id:
            raise ValueError("run_id cannot be empty")
        if not normalized_reason:
            raise ValueError("reason cannot be empty")
        if outcome not in {"CANCELLED", "SUPERSEDED"}:
            raise ValueError("terminal run outcome is invalid")

        sessions = [
            session
            for session in self._sessions.values()
            if session.run_id == normalized_run_id
        ]
        known_session_ids = {session.session_id for session in sessions}
        self.archive_root.mkdir(parents=True, exist_ok=True)
        for attempt_root in sorted(self.archive_root.iterdir()):
            if not attempt_root.is_dir():
                continue
            checkpoint = self.runtime_checkpoint_store.load(attempt_root)
            if (
                checkpoint is None
                or checkpoint.phase == "TERMINAL"
                or checkpoint.run_id != normalized_run_id
                or checkpoint.runtime_session_id in known_session_ids
            ):
                continue
            session, _ = self._restore_checkpoint_session(checkpoint)
            self._sessions[session.session_id] = session
            sessions.append(session)
            known_session_ids.add(session.session_id)

        return tuple(
            self._finish_terminated_session(
                session,
                reason=normalized_reason,
                outcome=outcome,
            )
            for session in sessions
        )

    def _restore_checkpoint_session(
        self,
        checkpoint: CodeRuntimeCheckpoint,
    ) -> tuple[_CodeRuntimeSession, bool]:
        recovery = checkpoint.recovery_state
        if recovery is None:
            raise RuntimeError(
                "Checkpoint predates resumable orchestration state; snapshots "
                "were preserved but automatic continuation is unsafe"
            )
        if checkpoint.phase not in {
            "CANDIDATE_READY",
            "AWAITING_SCHEDULER",
            "RECOVERY_REQUIRED",
        }:
            raise RuntimeError(
                f"Checkpoint phase {checkpoint.phase} is not an automatic "
                "startup recovery boundary"
            )
        resume_phase: CodeRuntimePhase = checkpoint.phase
        if checkpoint.phase == "RECOVERY_REQUIRED":
            if (
                recovery.review_loop.status == "REVIEWING"
                and checkpoint.candidate_snapshot_path
            ):
                resume_phase = "CANDIDATE_READY"
            elif recovery.review_loop.status == "ESCALATED_TO_SCHEDULER":
                resume_phase = "AWAITING_SCHEDULER"
            else:
                raise RuntimeError(
                    "RECOVERY_REQUIRED does not contain a safe role boundary"
                )
        if resume_phase != "CANDIDATE_READY" and (
            recovery.review_loop.status != "ESCALATED_TO_SCHEDULER"
        ):
            raise RuntimeError(
                "Only CANDIDATE_READY or a session awaiting Scheduler control "
                "can be resumed automatically"
            )
        if resume_phase == "CANDIDATE_READY" and (
            recovery.review_loop.status != "REVIEWING"
        ):
            raise RuntimeError(
                "CANDIDATE_READY must resume an unreviewed candidate"
            )

        attempt_root = Path(checkpoint.attempt_root).resolve()
        if not attempt_root.is_relative_to(self.archive_root):
            raise RuntimeError("Checkpoint attempt root is not archive-owned")
        run_layout = initialize_run_workspace(
            checkpoint.run_id,
            storage_root=self.run_storage_root,
        )
        integration_root = Path(checkpoint.integration_root).resolve()
        if integration_root != run_layout.integration_root.resolve():
            raise RuntimeError("Checkpoint integration root is not run-owned")
        integration_repository = IntegrationRepository(
            run_id=checkpoint.run_id,
            root=integration_root,
            receipts_root=run_layout.receipts_root,
        )
        status = integration_repository.status()
        if status.head_commit != checkpoint.integration_head_commit:
            raise RuntimeError(
                "Integration HEAD changed after the runtime checkpoint"
            )
        if not status.working_tree_clean:
            raise RuntimeError(
                "Integration tree has an uncommitted publish; reconciliation "
                "is required before automatic recovery"
            )

        def snapshot(relative: str | None) -> Path | None:
            if not relative:
                return None
            resolved = (attempt_root / relative).resolve()
            if not resolved.is_relative_to(attempt_root):
                raise RuntimeError("Checkpoint snapshot escaped its attempt root")
            return resolved

        pair, rebuilt = self.sandbox_manager.recover_pair(
            checkpoint.sandbox.to_pair(),
            candidate_snapshot=snapshot(checkpoint.candidate_snapshot_path),
            reviewer_snapshot=snapshot(checkpoint.reviewer_snapshot_path),
        )
        worker = self.worker_factory(
            self.model,
            **({"summary_model": self.summary_model} if self.summary_model is not None else {}),
            tools=self.tools,
            backend=self.sandbox_manager.worker_backend(pair),
            progress_every_tool_calls=self.progress_every_tool_calls,
            schema_repair_max_rounds=self.schema_repair_max_rounds,
            middleware=self._middleware(),
            checkpointer=self.checkpointer,
            store=self.store,
        )
        reviewer = self.reviewer_factory(
            self.reviewer_model,
            **({"summary_model": self.reviewer_summary_model} if self.reviewer_summary_model is not None else {}),
            tools=self.reviewer_tools,
            backend=self.sandbox_manager.reviewer_backend(pair),
            schema_repair_max_rounds=self.schema_repair_max_rounds,
            middleware=self._middleware(),
            checkpointer=self.checkpointer,
            store=self.store,
        )
        if not isinstance(recovery.worker_submission, dict):
            raise RuntimeError("Recovered CODE session has no Worker submission")
        if resume_phase != "CANDIDATE_READY" and recovery.last_report is None:
            raise RuntimeError("Recovered CODE session has no Reviewer report")
        session = _CodeRuntimeSession(
            session_id=checkpoint.runtime_session_id,
            run_id=checkpoint.run_id,
            run_layout=run_layout,
            contract=recovery.contract,
            candidate=recovery.candidate,
            loop=recovery.review_loop,
            pair=pair,
            worker=worker,
            reviewer=reviewer,
            worker_thread_id=checkpoint.worker_checkpoint_id,
            reviewer_thread_id=checkpoint.reviewer_checkpoint_id,
            integration_root=integration_root,
            integration_repository=integration_repository,
            base_revision=checkpoint.base_revision,
            archive_id=attempt_root.name,
            attempt_root=attempt_root,
            started_at=recovery.started_at,
            parent_attempt_id=recovery.parent_attempt_id,
            worker_id=recovery.worker_id,
            worker_submission=recovery.worker_submission,
            latest_candidate_relative=checkpoint.candidate_snapshot_path or "",
            last_review_details=recovery.last_review_details,
            last_report=recovery.last_report,
            superseded_attempt_records=list(
                recovery.superseded_attempt_records
            ),
            runtime_generation=checkpoint.generation,
            latest_reviewer_checkpoint_relative=(
                checkpoint.reviewer_snapshot_path
            ),
            recovery_phase=resume_phase,
        )
        self._commit_session_checkpoint(
            session,
            phase=resume_phase,
            reason=(
                "Startup recovery rebuilt the in-memory session and froze its "
                "Docker pair before new work was accepted."
            ),
        )
        return session, rebuilt

    @staticmethod
    def _preemption_policy(phase: CodeRuntimePhase) -> str:
        # Reviewers are deliberately short, non-preemptible critical sections.
        # An external control event may be recorded while they run, but it is
        # applied only after the complete Reviewer role turn returns.
        return (
            "DEFER_UNTIL_ROLE_BOUNDARY"
            if phase == "REVIEWER_RUNNING"
            else "SAFE_POINT"
        )

    @operation('Code Runtime / Save Checkpoint', fields=('candidate', 'phase', 'reason', 'generation'))
    def _commit_runtime_checkpoint(
        self,
        *,
        runtime_session_id: str,
        run_id: str,
        candidate: CodeCandidateRef,
        generation: int,
        phase: CodeRuntimePhase,
        pair: CodeSandboxPair,
        worker_thread_id: str,
        reviewer_thread_id: str,
        integration_root: Path,
        integration_repository: IntegrationRepository,
        base_revision: str,
        attempt_root: Path,
        candidate_snapshot_path: str | None = None,
        reviewer_snapshot_path: str | None = None,
        recovery_state: CodeRuntimeRecoveryState | None = None,
        cleanup_authorized: bool = False,
        reason: str = "",
    ) -> CodeRuntimeCheckpoint:
        """Commit a JSON-safe role/resource boundary outside LangGraph."""

        integration_status = integration_repository.status()
        checkpoint = CodeRuntimeCheckpoint(
            checkpoint_id=f"code-runtime-{uuid4().hex}",
            runtime_session_id=runtime_session_id,
            run_id=run_id,
            step_id=candidate.step_id,
            attempt_id=candidate.attempt_id,
            generation=generation,
            phase=phase,
            preemption_policy=self._preemption_policy(phase),
            worker_checkpoint_id=worker_thread_id,
            reviewer_checkpoint_id=reviewer_thread_id,
            integration_root=str(integration_root),
            integration_head_commit=integration_status.head_commit,
            base_revision=base_revision,
            attempt_root=str(attempt_root),
            candidate_snapshot_path=(candidate_snapshot_path or None),
            reviewer_snapshot_path=(reviewer_snapshot_path or None),
            sandbox=CodeSandboxPairRecord.from_pair(pair),
            recovery_state=recovery_state,
            cleanup_authorized=cleanup_authorized,
            reason=reason,
            committed_at=datetime.now(timezone.utc),
        )
        return self.runtime_checkpoint_store.commit(attempt_root, checkpoint)

    def _cleanup_committed_pair(
        self,
        *,
        pair: CodeSandboxPair,
        attempt_root: Path,
    ) -> None:
        """Delete Docker resources only after reading back terminal evidence."""

        self.runtime_checkpoint_store.require_cleanup_authorized(
            attempt_root,
            pair_id=pair.pair_id,
        )
        self.sandbox_manager.cleanup(pair)

    @staticmethod
    def _recovery_state(
        *,
        contract: CodeTaskContract,
        candidate: CodeCandidateRef,
        loop: CodeReviewLoopState,
        worker_submission: dict[str, Any] | None,
        review_details: dict[str, Any] | None,
        report: CodeReviewReport | None,
        started_at: datetime,
        parent_attempt_id: str | None,
        worker_id: str,
        superseded_attempt_records: Sequence[dict[str, Any]],
    ) -> CodeRuntimeRecoveryState:
        durable_review_details = None
        if review_details is not None:
            durable_review_details = {
                key: review_details.get(key)
                for key in (
                    "code_artifact_manifest",
                    "code_publication_receipt",
                )
                if review_details.get(key) is not None
            }
        return CodeRuntimeRecoveryState(
            contract=contract,
            candidate=candidate,
            review_loop=loop,
            worker_submission=worker_submission,
            last_review_details=durable_review_details,
            last_report=report,
            started_at=started_at,
            parent_attempt_id=parent_attempt_id,
            worker_id=worker_id,
            superseded_attempt_records=tuple(superseded_attempt_records),
        )

    def _middleware(self) -> list[Any]:
        if self.middleware_factory is None:
            return []
        return list(self.middleware_factory())

    @staticmethod
    def _identity(input_state: dict[str, Any], thread_id: str) -> CodeCandidateRef:
        event_id = str(input_state.get("event_id") or thread_id)
        raw_step_id = input_state.get("step_id", 1)
        step_id = int(raw_step_id)
        workspace_id = str(
            input_state.get("worker_id")
            or f"code:{event_id}:step:{step_id}"
        )
        return CodeCandidateRef(
            event_id=event_id,
            step_id=step_id,
            attempt_id=thread_id,
            workspace_id=workspace_id,
            candidate_revision=1,
        )

    def _attempt_root(self, candidate: CodeCandidateRef) -> tuple[str, Path]:
        archive_id = f"code-archive-{uuid4().hex}"
        root = self.archive_root / archive_id
        root.mkdir(parents=True, exist_ok=False)
        return archive_id, root

    @staticmethod
    def _used(details: dict[str, Any], name: str) -> int:
        summary = details.get("execution_summary", {})
        return max(int(summary.get(name, 0) or 0), 0)

    async def _invoke(
        self,
        agent,
        instruction: str,
        *,
        thread_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        # Fresh role invocation, retaining dialogue but not an earlier repair's stop flags.
        state = {**state, "worker_finalize_requested": False,
                 "worker_review_requested": False,
                 "worker_finalize_reason": "", "worker_finalization_model_calls_used": 0,
                 "worker_finalization_tool_calls_used": 0,
                 "worker_finalization_model_run_limit": WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS,
                 "worker_schema_repair_model_calls_used": 0,
                 "worker_schema_repair_model_run_limit": getattr(self, "schema_repair_max_rounds", 3)}
        if not thread_id.endswith(":reviewer"):
            role_budget = ""  # Actual Worker capacity is injected after preparation.
        else:
            role_budget = ("\n\n[本次审核可用额度]\n"
                           f"最多模型轮次：{state.get('executor_model_run_limit', 0)}；"
                           f"最多工具调用：{state.get('executor_tool_run_limit', 0)}。请依据实际证据提交报告。")
        details = await ask_worker(
            agent,
            instruction + role_budget,
            thread_id=thread_id,
            trace_role="reviewer" if thread_id.endswith(":reviewer") else "code",
            state_update=state,
            return_details=True,
        )
        if not isinstance(details, dict):
            raise RuntimeError("CODE Agent did not return structured details")
        return details

    async def ainvoke(
        self,
        input_state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        thread_id = str(configurable.get("thread_id") or "code-step")
        pause_control = configurable.get("event_pause_control")
        async with self._writer_slots:
            return await self._run(
                input_state,
                thread_id=thread_id,
                pause_control=pause_control,
            )

    @operation('Code Runtime / Attempt', fields=('input_state', 'thread_id'))
    async def _run(
        self,
        input_state: dict[str, Any],
        *,
        thread_id: str,
        pause_control=None,
    ) -> dict[str, Any]:
        raw_scheduler_decision = input_state.get("code_scheduler_decision")
        if raw_scheduler_decision is not None:
            session_id = str(
                input_state.get("code_runtime_session_id") or ""
            ).strip()
            if not session_id:
                raise ValueError(
                    "Scheduler CODE decision requires code_runtime_session_id"
                )
            return await self._resume_session(
                input_state,
                thread_id=thread_id,
                session_id=session_id,
                decision=SchedulerCodeDecision.model_validate(
                    raw_scheduler_decision
                ),
                pause_control=pause_control,
            )

        run_id = str(
            input_state.get("event_id")
            or input_state.get("planning_run_id")
            or thread_id
        ).strip()
        interrupted_session = next(
            (
                session
                for session in self._sessions.values()
                if session.run_id == run_id
                and session.candidate.attempt_id == thread_id
                and session.recovery_phase == "CANDIDATE_READY"
            ),
            None,
        )
        if interrupted_session is not None:
            return await self._resume_candidate_review(
                input_state,
                session=interrupted_session,
            )

        if self._sessions:
            raise RuntimeError(
                "A CODE pair is awaiting Scheduler control; another writer "
                "cannot start until that pair is continued, restarted, or stopped"
            )

        raw_contract = input_state.get("code_task")
        if raw_contract is None:
            raise ValueError("CODE Step is missing its frozen code_task contract")
        contract = CodeTaskContract.model_validate(raw_contract)

        messages = list(input_state.get("messages", []))
        initial_instruction = (
            _message_text(messages[-1]).strip()
            if messages
            else "Implement the frozen CODE contract."
        )
        candidate = self._identity(input_state, thread_id)
        run_layout = initialize_run_workspace(
            run_id,
            storage_root=self.run_storage_root,
        )
        worker_thread_id = f"{thread_id}:worker"
        reviewer_thread_id = f"{thread_id}:reviewer"
        loop = create_code_review_loop(
            candidate=candidate,
            worker_checkpoint_id=worker_thread_id,
            reviewer_checkpoint_id=reviewer_thread_id,
            max_repair_rounds=self.max_repair_rounds,
        )
        started_at = datetime.now(timezone.utc)
        parent_attempt_id = (
            str(input_state.get("code_parent_attempt_id") or "").strip()
            or None
        )
        superseded_attempt_records = list(
            input_state.get("code_superseded_attempt_records") or []
        )
        source_root = Path(
            str(input_state.get("conversation_workspace_root") or self.source_root)
        ).resolve()
        integration_root = initialize_code_integration(
            layout=run_layout,
            source_root=source_root,
        )
        integration_repository = IntegrationRepository(
            run_id=run_id,
            root=integration_root,
            receipts_root=run_layout.receipts_root,
        )
        integration_repository.initialize()
        integration_status_before = integration_repository.status()
        if not integration_status_before.working_tree_clean:
            raise RuntimeError(
                "CODE integration contains an uncommitted prior publication; "
                "recovery is required before starting another Worker"
            )
        initial_instruction = (
            initial_instruction
            + "\n\n[Accepted Integration Git State]\n"
            + json.dumps(
                integration_status_before.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\nThis workspace is a Git copy of the accepted integration HEAD. "
            "Use git status, git log, git show, or a scoped git diff when "
            "history is relevant; do not load the full history by default."
        )
        base_revision = compute_code_tree_revision(integration_root)
        archive_id, attempt_root = self._attempt_root(candidate)
        runtime_session_id = f"code-session-{uuid4().hex}"
        runtime_generation = max(
            int(input_state.get("event_generation", 1) or 1),
            1,
        )
        pair: CodeSandboxPair | None = None
        latest_candidate_relative = ""
        latest_reviewer_checkpoint_relative: str | None = None
        current_runtime_phase: CodeRuntimePhase = "WORKER_RUNNING"
        cleanup_authorized = False
        all_messages: list[Any] = []
        model_calls_used = 0
        tool_calls_used = 0
        show_all_toolsets_calls_used = 0
        worker_submission: dict[str, Any] | None = None
        review_details: dict[str, Any] = {}
        total_model_limit = max(
            int(input_state.get("executor_model_run_limit", 0) or 0),
            0,
        )
        total_tool_limit = max(
            int(input_state.get("executor_tool_run_limit", 0) or 0),
            0,
        )
        total_show_all_toolsets_limit = max(
            int(input_state.get("show_all_toolsets_run_limit", 0) or 0),
            0,
        )

        def budgeted_state(
            value: dict[str, Any],
            *,
            reserve_model: int = 0,
            reserve_tools: int = 0,
        ) -> dict[str, Any]:
            remaining_model = total_model_limit - model_calls_used
            remaining_tools = total_tool_limit - tool_calls_used
            available_model = remaining_model - reserve_model
            available_tools = remaining_tools - reserve_tools
            if available_model < 1:
                raise RuntimeError(
                    "CODE Step model budget cannot reach the next role while "
                    "preserving its review reserve"
                )
            if available_tools < 1:
                raise RuntimeError(
                    "CODE Step tool budget cannot reach the next role while "
                    "preserving its review reserve"
                )
            return {
                **value,
                "executor_model_run_limit": available_model,
                "executor_tool_run_limit": available_tools,
                # Worker and Reviewer share the Step-level discovery allowance.
                "show_all_toolsets_run_limit": max(
                    0, total_show_all_toolsets_limit - show_all_toolsets_calls_used
                ),
            }

        common_state = {
            key: value
            for key, value in input_state.items()
            if key != "messages"
        }
        common_state.update(
            {
                "code_task": contract.model_dump(mode="json"),
                "code_candidate": candidate.model_dump(mode="json"),
                "code_review_loop": loop.model_dump(mode="json"),
                "code_integration_status": integration_status_before.model_dump(
                    mode="json"
                ),
                "code_agent_finished": False,
            }
        )

        try:
            pair = self.sandbox_manager.create_pair(
                candidate.workspace_id,
                handoff_root=run_layout.handoff_root,
            )
            pair = self.sandbox_manager.copy_source(pair, integration_root)
            self._commit_runtime_checkpoint(
                runtime_session_id=runtime_session_id,
                run_id=run_id,
                candidate=candidate,
                generation=runtime_generation,
                phase="WORKER_RUNNING",
                pair=pair,
                worker_thread_id=worker_thread_id,
                reviewer_thread_id=reviewer_thread_id,
                integration_root=integration_root,
                integration_repository=integration_repository,
                base_revision=base_revision,
                attempt_root=attempt_root,
                reason="Code Worker owns the next complete role turn.",
            )
            worker = self.worker_factory(
                self.model,
                **({"summary_model": self.summary_model} if self.summary_model is not None else {}),
                tools=self.tools,
                backend=self.sandbox_manager.worker_backend(pair),
                progress_every_tool_calls=self.progress_every_tool_calls,
                schema_repair_max_rounds=self.schema_repair_max_rounds,
                middleware=self._middleware(),
                checkpointer=self.checkpointer,
                store=self.store,
            )
            reviewer = self.reviewer_factory(
                self.reviewer_model,
                **({"summary_model": self.reviewer_summary_model} if self.reviewer_summary_model is not None else {}),
                tools=self.reviewer_tools,
                backend=self.sandbox_manager.reviewer_backend(pair),
                schema_repair_max_rounds=self.schema_repair_max_rounds,
                middleware=self._middleware(),
                checkpointer=self.checkpointer,
                store=self.store,
            )

            worker_details = await self._invoke(
                worker,
                initial_instruction,
                thread_id=worker_thread_id,
                # Keep enough shared Step budget for the independent
                # Reviewer to test, publish, and submit its report.
                state=budgeted_state(
                    common_state,
                    reserve_model=3,
                    reserve_tools=3,
                ),
            )
            all_messages.extend(worker_details.get("current_turn_messages", []))
            model_calls_used += self._used(worker_details, "model_call_count")
            tool_calls_used += self._used(worker_details, "tool_call_count")
            show_all_toolsets_calls_used += self._used(
                worker_details,
                "show_all_toolsets_call_count",
            )
            worker_submission = worker_details.get("code_worker_submission")
            if not isinstance(worker_submission, dict):
                from workers.code_finalization import missing_handoff_message
                raise RuntimeError(missing_handoff_message(worker_details, "INITIAL_SUBMISSION"))
            loop = CodeReviewLoopState.model_validate(
                worker_details.get("code_review_loop")
                or loop.model_dump(mode="json")
            )
            candidate = CodeCandidateRef.model_validate(
                worker_details.get("code_candidate")
                or worker_submission.get("submission", {}).get("candidate")
                or candidate.model_dump(mode="json")
            )

            for review_pass in range(self.max_repair_rounds + 1):
                pair = self.sandbox_manager.handoff(pair, "REVIEWER")
                latest_candidate_relative = (
                    f"candidate/revision-{candidate.candidate_revision}-"
                    f"review-{review_pass + 1}"
                )
                candidate_snapshot = attempt_root / latest_candidate_relative
                self.sandbox_manager.export_candidate(pair, candidate_snapshot)
                if pause_control is not None and pause_control.requested:
                    pair = self.sandbox_manager.freeze(pair)
                    self._commit_runtime_checkpoint(
                        runtime_session_id=runtime_session_id,
                        run_id=run_id,
                        candidate=candidate,
                        generation=runtime_generation,
                        phase="CANDIDATE_READY",
                        pair=pair,
                        worker_thread_id=worker_thread_id,
                        reviewer_thread_id=reviewer_thread_id,
                        integration_root=integration_root,
                        integration_repository=integration_repository,
                        base_revision=base_revision,
                        attempt_root=attempt_root,
                        candidate_snapshot_path=latest_candidate_relative,
                        recovery_state=self._recovery_state(
                            contract=contract,
                            candidate=candidate,
                            loop=loop,
                            worker_submission=worker_submission,
                            review_details=None,
                            report=None,
                            started_at=started_at,
                            parent_attempt_id=parent_attempt_id,
                            worker_id=candidate.workspace_id,
                            superseded_attempt_records=(
                                superseded_attempt_records
                            ),
                        ),
                        reason="INSERT paused Code after a complete Worker turn.",
                    )
                    await pause_control.pause_point(
                        on_pause=self._writer_slots.release,
                        on_resume=self._writer_slots.acquire,
                    )
                    pair = self.sandbox_manager.handoff(pair, "REVIEWER")
                current_runtime_phase = "REVIEWER_RUNNING"
                self._commit_runtime_checkpoint(
                    runtime_session_id=runtime_session_id,
                    run_id=run_id,
                    candidate=candidate,
                    generation=runtime_generation,
                    phase=current_runtime_phase,
                    pair=pair,
                    worker_thread_id=worker_thread_id,
                    reviewer_thread_id=reviewer_thread_id,
                    integration_root=integration_root,
                    integration_repository=integration_repository,
                    base_revision=base_revision,
                    attempt_root=attempt_root,
                    candidate_snapshot_path=latest_candidate_relative,
                    reason=(
                        "Reviewer role turn is non-preemptible; external control "
                        "is deferred until this invocation returns."
                    ),
                )
                publication_context = CodePublicationContext(
                    candidate_root=str(candidate_snapshot),
                    target_root=str(integration_root),
                    base_revision=base_revision,
                )
                reviewer_state = {
                    "code_tool_audit": _review_tool_audit({"messages": all_messages})[0],
                    **common_state,
                    "code_candidate": candidate.model_dump(mode="json"),
                    "code_review_loop": loop.model_dump(mode="json"),
                    "code_worker_submission": worker_submission,
                    "code_publication_context": publication_context.model_dump(
                        mode="json"
                    ),
                    "code_agent_finished": False,
                }
                review_details = await self._invoke(
                    reviewer,
                    render_prompt(
                        "runtime/code_review_handoff",
                        review_pass=review_pass + 1,
                    ),
                    thread_id=reviewer_thread_id,
                    state=budgeted_state(reviewer_state),
                )
                all_messages.extend(
                    review_details.get("current_turn_messages", [])
                )
                model_calls_used += self._used(
                    review_details,
                    "model_call_count",
                )
                tool_calls_used += self._used(
                    review_details,
                    "tool_call_count",
                )
                show_all_toolsets_calls_used += self._used(
                    review_details,
                    "show_all_toolsets_call_count",
                )
                loop = CodeReviewLoopState.model_validate(
                    review_details.get("code_review_loop")
                )
                candidate = loop.candidate
                current_runtime_phase = "REVIEW_COMPLETED"
                boundary_report = CodeReviewReport.model_validate(
                    review_details.get("code_review_report")
                )
                self._commit_runtime_checkpoint(
                    runtime_session_id=runtime_session_id,
                    run_id=run_id,
                    candidate=candidate,
                    generation=runtime_generation,
                    phase=current_runtime_phase,
                    pair=pair,
                    worker_thread_id=worker_thread_id,
                    reviewer_thread_id=reviewer_thread_id,
                    integration_root=integration_root,
                    integration_repository=integration_repository,
                    base_revision=base_revision,
                    attempt_root=attempt_root,
                    candidate_snapshot_path=latest_candidate_relative,
                    recovery_state=self._recovery_state(
                        contract=contract,
                        candidate=candidate,
                        loop=loop,
                        worker_submission=worker_submission,
                        review_details=review_details,
                        report=boundary_report,
                        started_at=started_at,
                        parent_attempt_id=parent_attempt_id,
                        worker_id=candidate.workspace_id,
                        superseded_attempt_records=superseded_attempt_records,
                    ),
                    reason="Reviewer completed its indivisible role turn.",
                )
                if loop.status in {"APPLIED", "ESCALATED_TO_SCHEDULER"}:
                    break
                if loop.status != "WAITING_FOR_WORKER":
                    raise RuntimeError(
                        "Code Reviewer returned an invalid loop state: "
                        f"{loop.status}"
                    )

                pair = self.sandbox_manager.handoff(pair, "WORKER")
                current_runtime_phase = "WORKER_RUNNING"
                self._commit_runtime_checkpoint(
                    runtime_session_id=runtime_session_id,
                    run_id=run_id,
                    candidate=candidate,
                    generation=runtime_generation,
                    phase=current_runtime_phase,
                    pair=pair,
                    worker_thread_id=worker_thread_id,
                    reviewer_thread_id=reviewer_thread_id,
                    integration_root=integration_root,
                    integration_repository=integration_repository,
                    base_revision=base_revision,
                    attempt_root=attempt_root,
                    candidate_snapshot_path=latest_candidate_relative,
                    reason="Reviewer handed one complete repair instruction to Worker.",
                )
                repair_state = {
                    **common_state,
                    "code_candidate": candidate.model_dump(mode="json"),
                    "code_review_loop": loop.model_dump(mode="json"),
                    "code_repair_instruction": (
                        loop.pending_instruction.model_dump(mode="json")
                        if loop.pending_instruction is not None
                        else None
                    ),
                    "code_worker_submission": worker_submission,
                    "code_agent_finished": False,
                }
                worker_details = await self._invoke(
                    worker,
                    render_prompt(
                        "runtime/code_repair_handoff",
                        repair_round=loop.repair_round,
                    ),
                    thread_id=worker_thread_id,
                    # A repair response is not useful unless the Reviewer can
                    # run again and issue the final structured decision.
                    state=budgeted_state(
                        repair_state,
                        reserve_model=2,
                        reserve_tools=2,
                    ),
                )
                all_messages.extend(
                    worker_details.get("current_turn_messages", [])
                )
                model_calls_used += self._used(
                    worker_details,
                    "model_call_count",
                )
                tool_calls_used += self._used(
                    worker_details,
                    "tool_call_count",
                )
                show_all_toolsets_calls_used += self._used(
                    worker_details,
                    "show_all_toolsets_call_count",
                )
                loop = CodeReviewLoopState.model_validate(
                    worker_details.get("code_review_loop")
                )
                if loop.status != "REVIEWING":
                    from workers.code_finalization import missing_handoff_message
                    raise RuntimeError(missing_handoff_message(worker_details, "REPAIR_RESPONSE"))
                candidate = loop.candidate
                updated_submission = worker_details.get(
                    "code_worker_submission"
                )
                if isinstance(updated_submission, dict):
                    worker_submission = updated_submission
            else:
                raise RuntimeError("Code review loop exceeded its repair bound")

            report = CodeReviewReport.model_validate(
                review_details.get("code_review_report")
            )
            if loop.status == "ESCALATED_TO_SCHEDULER":
                pair = self.sandbox_manager.freeze(pair)
                session = _CodeRuntimeSession(
                    session_id=runtime_session_id,
                    run_id=run_id,
                    run_layout=run_layout,
                    contract=contract,
                    candidate=candidate,
                    loop=loop,
                    pair=pair,
                    worker=worker,
                    reviewer=reviewer,
                    worker_thread_id=worker_thread_id,
                    reviewer_thread_id=reviewer_thread_id,
                    integration_root=integration_root,
                    integration_repository=integration_repository,
                    base_revision=base_revision,
                    archive_id=archive_id,
                    attempt_root=attempt_root,
                    started_at=started_at,
                    parent_attempt_id=parent_attempt_id,
                    worker_id=candidate.workspace_id,
                    worker_submission=worker_submission,
                    latest_candidate_relative=latest_candidate_relative,
                    last_review_details=review_details,
                    last_report=report,
                    all_messages=all_messages,
                    superseded_attempt_records=superseded_attempt_records,
                    runtime_generation=runtime_generation,
                )
                current_runtime_phase = "AWAITING_SCHEDULER"
                self._commit_session_checkpoint(
                    session,
                    phase=current_runtime_phase,
                    reason="Frozen pair is waiting for Scheduler control.",
                )
                self._sessions[runtime_session_id] = session
                # Ownership moves into _sessions; the finally block must not
                # destroy the containers or volumes that CONTINUE relies on.
                pair = None
                return {
                    "messages": [AIMessage(content=report.summary)],
                    "executor_model_calls_used": model_calls_used,
                    "executor_tool_calls_used": tool_calls_used,
                    "show_all_toolsets_calls_used": show_all_toolsets_calls_used,
                    "code_worker_submission": worker_submission,
                    "code_review_loop": loop.model_dump(mode="json"),
                    "code_review_report": report.model_dump(mode="json"),
                    "code_artifact_manifest": None,
                    "code_publication_receipt": None,
                    "code_handoff_publication_receipts": [],
                    "code_integration_commit": None,
                    "code_integration_status": (
                        integration_repository.status().model_dump(mode="json")
                    ),
                    "code_attempt_archive": None,
                    "code_attempt_final_record": None,
                    "code_runtime_session_id": runtime_session_id,
                    "code_superseded_attempt_records": (
                        session.superseded_attempt_records
                    ),
                    "code_agent_messages": all_messages,
                }
            pair = self.sandbox_manager.freeze(pair)
            reviewer_relative = "reviewer"
            self.sandbox_manager.export_review(
                pair,
                attempt_root / reviewer_relative,
            )
            preserved_at = datetime.now(timezone.utc)
            archive = CodeAttemptArchive(
                archive_id=archive_id,
                root_path=str(attempt_root),
                candidate_snapshot_path=latest_candidate_relative,
                reviewer_snapshot_path=reviewer_relative,
                preserved_at=preserved_at,
                retain_until=preserved_at
                + timedelta(minutes=self.archive_retention_minutes),
            )

            manifest_value = review_details.get("code_artifact_manifest")
            receipt_value = review_details.get("code_publication_receipt")
            final_record_value = None
            handoff_receipts: list[dict[str, Any]] = []
            integration_commit_value = None
            if loop.status == "APPLIED":
                manifest = CodeArtifactManifest.model_validate(manifest_value)
                receipt = CodePublicationReceipt.model_validate(receipt_value)
                test_summary = "; ".join(
                    f"{check.check_id}={check.status}: {check.summary}"
                    for check in report.check_results
                ) or report.verification_summary
                integration_commit = integration_repository.commit_accepted(
                    step_id=candidate.step_id,
                    candidate_revision=candidate.candidate_revision,
                    manifest_id=manifest.manifest_id,
                    publication_id=receipt.publication_id,
                    approved_paths=tuple(entry.path for entry in manifest.files),
                    review_summary=report.summary,
                    test_summary=test_summary,
                )
                integration_commit_value = integration_commit.model_dump(
                    mode="json"
                )
                for entry in manifest.files:
                    source = (integration_root / entry.path).resolve()
                    candidate_id = (
                        f"code-step-{candidate.step_id}-revision-"
                        f"{candidate.candidate_revision}:{entry.path}"
                    )
                    shared_candidate = ResolvedArtifactCandidate(
                        candidate_id=candidate_id,
                        review_ref=(
                            f"code/step-{candidate.step_id}/revision-"
                            f"{candidate.candidate_revision}/{entry.path}"
                        ),
                        kind="WORKSPACE_FILE",
                        description=(
                            "Code Reviewer approved file from the run integration tree."
                        ),
                        verified=True,
                        location=entry.path,
                        storage_path=str(source),
                        size_bytes=entry.size_bytes,
                        sha256=entry.sha256,
                    )
                    shared_receipt = publish_artifact_to_handoff(
                        layout=run_layout,
                        candidate=shared_candidate,
                    )
                    handoff_receipts.append(
                        shared_receipt.model_dump(mode="json")
                    )
                final_record = CodeAttemptFinalRecord(
                    record_id=f"code-final-{uuid4().hex}",
                    candidate=candidate,
                    parent_attempt_id=(
                        str(input_state.get("code_parent_attempt_id") or "").strip()
                        or None
                    ),
                    outcome="APPLIED",
                    terminal_reason=report.summary,
                    started_at=started_at,
                    finalized_at=datetime.now(timezone.utc),
                    worker_checkpoint_id=worker_thread_id,
                    reviewer_checkpoint_id=reviewer_thread_id,
                    docker_image=pair.image,
                    archive=archive,
                    artifact_manifest=manifest,
                    review_report=report,
                    publication=receipt,
                    integration_commit=integration_commit,
                )
                final_record_value = final_record.model_dump(mode="json")
                (attempt_root / "final_record.json").write_text(
                    json.dumps(final_record_value, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            current_runtime_phase = "TERMINAL"
            self._commit_runtime_checkpoint(
                runtime_session_id=runtime_session_id,
                run_id=run_id,
                candidate=candidate,
                generation=runtime_generation,
                phase=current_runtime_phase,
                pair=pair,
                worker_thread_id=worker_thread_id,
                reviewer_thread_id=reviewer_thread_id,
                integration_root=integration_root,
                integration_repository=integration_repository,
                base_revision=base_revision,
                attempt_root=attempt_root,
                candidate_snapshot_path=latest_candidate_relative,
                reviewer_snapshot_path=reviewer_relative,
                cleanup_authorized=True,
                reason="Host archive and terminal record are committed.",
            )
            cleanup_authorized = True

            final_text = report.summary
            return {
                "messages": [AIMessage(content=final_text)],
                "executor_model_calls_used": model_calls_used,
                "executor_tool_calls_used": tool_calls_used,
                "show_all_toolsets_calls_used": show_all_toolsets_calls_used,
                "code_worker_submission": worker_submission,
                "code_review_loop": loop.model_dump(mode="json"),
                "code_review_report": report.model_dump(mode="json"),
                "code_artifact_manifest": manifest_value,
                "code_publication_receipt": receipt_value,
                "code_handoff_publication_receipts": handoff_receipts,
                "code_integration_commit": integration_commit_value,
                "code_integration_status": integration_repository.status().model_dump(
                    mode="json"
                ),
                "code_attempt_archive": archive.model_dump(mode="json"),
                "code_attempt_final_record": final_record_value,
                "code_runtime_session_id": None,
                "code_superseded_attempt_records": list(
                    input_state.get("code_superseded_attempt_records") or []
                ),
                "code_agent_messages": all_messages,
            }
        except BaseException as error:
            if pair is not None and not cleanup_authorized:
                try:
                    pair = self.sandbox_manager.freeze(pair)
                    current_runtime_phase = "RECOVERY_REQUIRED"
                    recovery_state = None
                    if (
                        isinstance(worker_submission, dict)
                        and latest_candidate_relative
                    ):
                        recovery_state = self._recovery_state(
                            contract=contract,
                            candidate=candidate,
                            loop=loop,
                            worker_submission=worker_submission,
                            review_details=(review_details or None),
                            report=(
                                CodeReviewReport.model_validate(
                                    review_details.get("code_review_report")
                                )
                                if review_details.get("code_review_report")
                                is not None
                                else None
                            ),
                            started_at=started_at,
                            parent_attempt_id=parent_attempt_id,
                            worker_id=candidate.workspace_id,
                            superseded_attempt_records=(
                                superseded_attempt_records
                            ),
                        )
                    self._commit_runtime_checkpoint(
                        runtime_session_id=runtime_session_id,
                        run_id=run_id,
                        candidate=candidate,
                        generation=runtime_generation,
                        phase=current_runtime_phase,
                        pair=pair,
                        worker_thread_id=worker_thread_id,
                        reviewer_thread_id=reviewer_thread_id,
                        integration_root=integration_root,
                        integration_repository=integration_repository,
                        base_revision=base_revision,
                        attempt_root=attempt_root,
                        candidate_snapshot_path=(latest_candidate_relative or None),
                        reviewer_snapshot_path=latest_reviewer_checkpoint_relative,
                        recovery_state=recovery_state,
                        reason=f"Runtime stopped before cleanup authorization: {type(error).__name__}",
                    )
                except Exception:
                    # Preserve the original failure.  Most importantly, do not
                    # delete a pair whose recovery record could not be updated.
                    pass
            raise
        finally:
            if pair is not None:
                if cleanup_authorized:
                    self._cleanup_committed_pair(
                        pair=pair,
                        attempt_root=attempt_root,
                    )

    def _session_state(
        self,
        input_state: dict[str, Any],
        session: _CodeRuntimeSession,
    ) -> dict[str, Any]:
        ignored = {
            "messages",
            "code_scheduler_decision",
            "code_runtime_session_id",
        }
        state = {
            key: value
            for key, value in input_state.items()
            if key not in ignored
        }
        state.update(
            {
                "worker_id": session.worker_id,
                "code_task": session.contract.model_dump(mode="json"),
                "code_candidate": session.candidate.model_dump(mode="json"),
                "code_review_loop": session.loop.model_dump(mode="json"),
                "code_worker_submission": session.worker_submission,
                "code_integration_status": (
                    session.integration_repository.status().model_dump(mode="json")
                ),
                "code_agent_finished": False,
            }
        )
        return state

    def _commit_session_checkpoint(
        self,
        session: _CodeRuntimeSession,
        *,
        phase: CodeRuntimePhase,
        cleanup_authorized: bool = False,
        reason: str,
    ) -> CodeRuntimeCheckpoint:
        return self._commit_runtime_checkpoint(
            runtime_session_id=session.session_id,
            run_id=session.run_id,
            candidate=session.candidate,
            generation=session.runtime_generation,
            phase=phase,
            pair=session.pair,
            worker_thread_id=session.worker_thread_id,
            reviewer_thread_id=session.reviewer_thread_id,
            integration_root=session.integration_root,
            integration_repository=session.integration_repository,
            base_revision=session.base_revision,
            attempt_root=session.attempt_root,
            candidate_snapshot_path=(session.latest_candidate_relative or None),
            reviewer_snapshot_path=(
                session.latest_reviewer_checkpoint_relative
            ),
            recovery_state=self._recovery_state(
                contract=session.contract,
                candidate=session.candidate,
                loop=session.loop,
                worker_submission=session.worker_submission,
                review_details=session.last_review_details,
                report=session.last_report,
                started_at=session.started_at,
                parent_attempt_id=session.parent_attempt_id,
                worker_id=session.worker_id,
                superseded_attempt_records=session.superseded_attempt_records,
            ),
            cleanup_authorized=cleanup_authorized,
            reason=reason,
        )

    def _paused_result(
        self,
        session: _CodeRuntimeSession,
        *,
        report: CodeReviewReport,
        usage: _InvocationUsage,
        scheduler_decision: SchedulerCodeDecision | None = None,
    ) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content=report.summary)],
            "executor_model_calls_used": usage.model_calls,
            "executor_tool_calls_used": usage.tool_calls,
            "show_all_toolsets_calls_used": usage.show_all_toolsets_calls,
            "code_worker_submission": session.worker_submission,
            "code_review_loop": session.loop.model_dump(mode="json"),
            "code_review_report": report.model_dump(mode="json"),
            "code_artifact_manifest": None,
            "code_publication_receipt": None,
            "code_handoff_publication_receipts": [],
            "code_integration_commit": None,
            "code_integration_status": (
                session.integration_repository.status().model_dump(mode="json")
            ),
            "code_attempt_archive": None,
            "code_attempt_final_record": None,
            "code_runtime_session_id": session.session_id,
            "code_scheduler_decision_applied": (
                scheduler_decision.model_dump(mode="json")
                if scheduler_decision is not None
                else None
            ),
            "code_superseded_attempt_records": session.superseded_attempt_records,
            "code_agent_messages": session.all_messages,
        }

    async def _resume_candidate_review(
        self,
        input_state: dict[str, Any],
        *,
        session: _CodeRuntimeSession,
    ) -> dict[str, Any]:
        """Continue a crash-recovered candidate at the Reviewer boundary."""

        usage = _InvocationUsage(
            model_limit=max(
                int(input_state.get("executor_model_run_limit", 0) or 0),
                0,
            ),
            tool_limit=max(
                int(input_state.get("executor_tool_run_limit", 0) or 0),
                0,
            ),
        )
        try:
            session.pair = self.sandbox_manager.handoff(
                session.pair,
                "REVIEWER",
            )
            candidate_snapshot = (
                session.attempt_root / session.latest_candidate_relative
            )
            self._commit_session_checkpoint(
                session,
                phase="REVIEWER_RUNNING",
                reason=(
                    "Crash recovery handed the preserved candidate to the "
                    "non-preemptible Reviewer."
                ),
            )
            reviewer_state = {
                    "code_tool_audit": _review_tool_audit({"messages": session.all_messages})[0],
                **self._session_state(input_state, session),
                "code_publication_context": CodePublicationContext(
                    candidate_root=str(candidate_snapshot),
                    target_root=str(session.integration_root),
                    base_revision=session.base_revision,
                ).model_dump(mode="json"),
            }
            review_details = await self._invoke(
                session.reviewer,
                render_prompt("runtime/code_review_handoff", review_pass=1),
                thread_id=session.reviewer_thread_id,
                state=usage.state(reviewer_state),
            )
            usage.add(review_details)
            session.all_messages.extend(
                review_details.get("current_turn_messages", [])
            )
            session.loop = CodeReviewLoopState.model_validate(
                review_details.get("code_review_loop")
            )
            session.candidate = session.loop.candidate
            report = CodeReviewReport.model_validate(
                review_details.get("code_review_report")
            )
            session.last_review_details = review_details
            session.last_report = report

            if session.loop.status == "APPLIED":
                return self._finish_applied_session(
                    session,
                    report=report,
                    usage=usage,
                    scheduler_decision=None,
                )
            if session.loop.status == "WAITING_FOR_WORKER":
                # Recovery deliberately avoids silently starting a new repair
                # cycle. The Scheduler receives the Reviewer's concise report
                # and decides whether preserving this attempt is still useful.
                session.loop = session.loop.model_copy(
                    update={
                        "status": "ESCALATED_TO_SCHEDULER",
                        "pending_instruction": None,
                        "pending_scheduler_directive": None,
                        "terminal_summary": report.summary,
                    }
                )
            if session.loop.status != "ESCALATED_TO_SCHEDULER":
                raise RuntimeError(
                    "Recovered Code Reviewer returned an invalid loop state: "
                    f"{session.loop.status}"
                )
            session.pair = self.sandbox_manager.freeze(session.pair)
            session.recovery_phase = "AWAITING_SCHEDULER"
            self._commit_session_checkpoint(
                session,
                phase="AWAITING_SCHEDULER",
                reason=(
                    "Recovered Reviewer result is waiting for an explicit "
                    "Scheduler decision."
                ),
            )
            self._sessions[session.session_id] = session
            return self._paused_result(session, report=report, usage=usage)
        except Exception as error:
            session.pair = self.sandbox_manager.freeze(session.pair)
            session.recovery_phase = "RECOVERY_REQUIRED"
            self._commit_session_checkpoint(
                session,
                phase="RECOVERY_REQUIRED",
                reason=(
                    "Recovered candidate failed before cleanup authorization: "
                    f"{type(error).__name__}"
                ),
            )
            raise

    @operation('Code Runtime / Archive', fields=())
    def _archive_session(
        self,
        session: _CodeRuntimeSession,
    ) -> CodeAttemptArchive:
        session.pair = self.sandbox_manager.freeze(session.pair)
        reviewer_relative = "reviewer"
        self.sandbox_manager.export_review(
            session.pair,
            session.attempt_root / reviewer_relative,
        )
        preserved_at = datetime.now(timezone.utc)
        return CodeAttemptArchive(
            archive_id=session.archive_id,
            root_path=str(session.attempt_root),
            candidate_snapshot_path=session.latest_candidate_relative,
            reviewer_snapshot_path=reviewer_relative,
            preserved_at=preserved_at,
            retain_until=(
                preserved_at + timedelta(minutes=self.archive_retention_minutes)
            ),
        )

    def _finish_cancelled_session(
        self,
        session: _CodeRuntimeSession,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Commit cancellation evidence before deleting a frozen Docker pair."""

        return self._finish_terminated_session(
            session,
            reason=reason,
            outcome="CANCELLED",
        )

    @operation('Code Runtime / Terminate', fields=())
    def _finish_terminated_session(
        self,
        session: _CodeRuntimeSession,
        *,
        reason: str,
        outcome: str,
    ) -> dict[str, Any]:
        """Archive one cancelled/superseded pair before Docker cleanup."""

        if outcome not in {"CANCELLED", "SUPERSEDED"}:
            raise ValueError("terminal session outcome is invalid")

        archive = self._archive_session(session)
        manifest = CodeArtifactManifest(
            manifest_id=f"{outcome.lower()}-{uuid4().hex[:24]}",
            candidate=session.candidate,
            files=(),
            created_at=datetime.now(timezone.utc),
        )
        final_record = CodeAttemptFinalRecord(
            record_id=f"code-final-{uuid4().hex}",
            candidate=session.candidate,
            parent_attempt_id=session.parent_attempt_id,
            outcome=outcome,
            terminal_reason=reason,
            failure_stage=None,
            started_at=session.started_at,
            finalized_at=datetime.now(timezone.utc),
            worker_checkpoint_id=session.worker_thread_id,
            reviewer_checkpoint_id=session.reviewer_thread_id,
            docker_image=session.pair.image,
            archive=archive,
            artifact_manifest=manifest,
            review_report=session.last_report,
        )
        record_value = final_record.model_dump(mode="json")
        (session.attempt_root / "final_record.json").write_text(
            json.dumps(record_value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        session.latest_reviewer_checkpoint_relative = (
            archive.reviewer_snapshot_path
        )
        self._commit_session_checkpoint(
            session,
            phase="TERMINAL",
            cleanup_authorized=True,
            reason=f"{outcome.title()} archive and terminal record are committed.",
        )
        self._sessions.pop(session.session_id, None)
        self._cleanup_committed_pair(
            pair=session.pair,
            attempt_root=session.attempt_root,
        )
        return record_value

    @operation('Code Runtime / Finish Unpublished Attempt', fields=())
    def _finish_unpublished_session(
        self,
        session: _CodeRuntimeSession,
        *,
        decision: SchedulerCodeDecision,
        outcome: str,
        failure_stage: str | None,
        usage: _InvocationUsage,
    ) -> dict[str, Any]:
        if session.last_report is None:
            raise RuntimeError("Cannot finalize CODE attempt without Reviewer report")
        archive = self._archive_session(session)
        manifest = CodeArtifactManifest(
            manifest_id=f"unpublished-{uuid4().hex[:24]}",
            candidate=session.candidate,
            files=(),
            created_at=datetime.now(timezone.utc),
        )
        final_record = CodeAttemptFinalRecord(
            record_id=f"code-final-{uuid4().hex}",
            candidate=session.candidate,
            parent_attempt_id=session.parent_attempt_id,
            outcome=outcome,
            terminal_reason=decision.reason,
            failure_stage=failure_stage,
            started_at=session.started_at,
            finalized_at=datetime.now(timezone.utc),
            worker_checkpoint_id=session.worker_thread_id,
            reviewer_checkpoint_id=session.reviewer_thread_id,
            docker_image=session.pair.image,
            archive=archive,
            artifact_manifest=manifest,
            review_report=session.last_report,
            scheduler_decision=decision,
        )
        record_value = final_record.model_dump(mode="json")
        (session.attempt_root / "final_record.json").write_text(
            json.dumps(record_value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        session.latest_reviewer_checkpoint_relative = (
            archive.reviewer_snapshot_path
        )
        self._commit_session_checkpoint(
            session,
            phase="TERMINAL",
            cleanup_authorized=True,
            reason="Unpublished archive and terminal record are committed.",
        )
        self._sessions.pop(session.session_id, None)
        self._cleanup_committed_pair(
            pair=session.pair,
            attempt_root=session.attempt_root,
        )
        return {
            "messages": [AIMessage(content=session.last_report.summary)],
            "executor_model_calls_used": usage.model_calls,
            "executor_tool_calls_used": usage.tool_calls,
            "show_all_toolsets_calls_used": usage.show_all_toolsets_calls,
            "code_worker_submission": session.worker_submission,
            "code_review_loop": session.loop.model_dump(mode="json"),
            "code_review_report": session.last_report.model_dump(mode="json"),
            "code_artifact_manifest": manifest.model_dump(mode="json"),
            "code_publication_receipt": None,
            "code_handoff_publication_receipts": [],
            "code_integration_commit": None,
            "code_integration_status": (
                session.integration_repository.status().model_dump(mode="json")
            ),
            "code_attempt_archive": archive.model_dump(mode="json"),
            "code_attempt_final_record": record_value,
            "code_runtime_session_id": None,
            "code_scheduler_decision_applied": decision.model_dump(mode="json"),
            "code_superseded_attempt_records": session.superseded_attempt_records,
            "code_agent_messages": session.all_messages,
        }

    @operation('Code Runtime / Finish Published Attempt', fields=())
    def _finish_applied_session(
        self,
        session: _CodeRuntimeSession,
        *,
        report: CodeReviewReport,
        usage: _InvocationUsage,
        scheduler_decision: SchedulerCodeDecision | None,
    ) -> dict[str, Any]:
        review_details = session.last_review_details or {}
        manifest = CodeArtifactManifest.model_validate(
            review_details.get("code_artifact_manifest")
        )
        receipt = CodePublicationReceipt.model_validate(
            review_details.get("code_publication_receipt")
        )
        test_summary = "; ".join(
            f"{check.check_id}={check.status}: {check.summary}"
            for check in report.check_results
        ) or report.verification_summary
        integration_commit = session.integration_repository.commit_accepted(
            step_id=session.candidate.step_id,
            candidate_revision=session.candidate.candidate_revision,
            manifest_id=manifest.manifest_id,
            publication_id=receipt.publication_id,
            approved_paths=tuple(entry.path for entry in manifest.files),
            review_summary=report.summary,
            test_summary=test_summary,
        )
        handoff_receipts: list[dict[str, Any]] = []
        for entry in manifest.files:
            source = (session.integration_root / entry.path).resolve()
            shared_candidate = ResolvedArtifactCandidate(
                candidate_id=(
                    f"code-step-{session.candidate.step_id}-revision-"
                    f"{session.candidate.candidate_revision}:{entry.path}"
                ),
                review_ref=(
                    f"code/step-{session.candidate.step_id}/revision-"
                    f"{session.candidate.candidate_revision}/{entry.path}"
                ),
                kind="WORKSPACE_FILE",
                description=(
                    "Code Reviewer approved file from the run integration tree."
                ),
                verified=True,
                location=entry.path,
                storage_path=str(source),
                size_bytes=entry.size_bytes,
                sha256=entry.sha256,
            )
            handoff_receipts.append(
                publish_artifact_to_handoff(
                    layout=session.run_layout,
                    candidate=shared_candidate,
                ).model_dump(mode="json")
            )
        archive = self._archive_session(session)
        final_record = CodeAttemptFinalRecord(
            record_id=f"code-final-{uuid4().hex}",
            candidate=session.candidate,
            parent_attempt_id=session.parent_attempt_id,
            outcome="APPLIED",
            terminal_reason=report.summary,
            started_at=session.started_at,
            finalized_at=datetime.now(timezone.utc),
            worker_checkpoint_id=session.worker_thread_id,
            reviewer_checkpoint_id=session.reviewer_thread_id,
            docker_image=session.pair.image,
            archive=archive,
            artifact_manifest=manifest,
            review_report=report,
            publication=receipt,
            integration_commit=integration_commit,
        )
        record_value = final_record.model_dump(mode="json")
        (session.attempt_root / "final_record.json").write_text(
            json.dumps(record_value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        session.latest_reviewer_checkpoint_relative = (
            archive.reviewer_snapshot_path
        )
        self._commit_session_checkpoint(
            session,
            phase="TERMINAL",
            cleanup_authorized=True,
            reason="Applied archive, Git commit, and terminal record are committed.",
        )
        self._sessions.pop(session.session_id, None)
        self._cleanup_committed_pair(
            pair=session.pair,
            attempt_root=session.attempt_root,
        )
        return {
            "messages": [AIMessage(content=report.summary)],
            "executor_model_calls_used": usage.model_calls,
            "executor_tool_calls_used": usage.tool_calls,
            "show_all_toolsets_calls_used": usage.show_all_toolsets_calls,
            "code_worker_submission": session.worker_submission,
            "code_review_loop": session.loop.model_dump(mode="json"),
            "code_review_report": report.model_dump(mode="json"),
            "code_artifact_manifest": manifest.model_dump(mode="json"),
            "code_publication_receipt": receipt.model_dump(mode="json"),
            "code_handoff_publication_receipts": handoff_receipts,
            "code_integration_commit": integration_commit.model_dump(mode="json"),
            "code_integration_status": (
                session.integration_repository.status().model_dump(mode="json")
            ),
            "code_attempt_archive": archive.model_dump(mode="json"),
            "code_attempt_final_record": record_value,
            "code_runtime_session_id": None,
            "code_scheduler_decision_applied": (
                scheduler_decision.model_dump(mode="json")
                if scheduler_decision is not None
                else None
            ),
            "code_superseded_attempt_records": session.superseded_attempt_records,
            "code_agent_messages": session.all_messages,
        }

    @operation('Code Runtime / Resume', fields=())
    async def _resume_session(
        self,
        input_state: dict[str, Any],
        *,
        thread_id: str,
        session_id: str,
        decision: SchedulerCodeDecision,
        pause_control=None,
    ) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError(
                "The CODE runtime session is unavailable; it may have already "
                "terminated or the process may have restarted"
            )
        if session.loop.status != "ESCALATED_TO_SCHEDULER":
            raise RuntimeError("CODE session is not awaiting Scheduler control")
        session.loop = apply_scheduler_code_decision(session.loop, decision)
        usage = _InvocationUsage(
            model_limit=max(
                int(input_state.get("executor_model_run_limit", 0) or 0),
                0,
            ),
            tool_limit=max(
                int(input_state.get("executor_tool_run_limit", 0) or 0),
                0,
            ),
        )

        if decision.action == "STOP":
            return self._finish_unpublished_session(
                session,
                decision=decision,
                outcome="FAILED",
                failure_stage="REVIEWER",
                usage=usage,
            )

        if decision.action == "RESTART":
            old_attempt_id = session.candidate.attempt_id
            terminal = self._finish_unpublished_session(
                session,
                decision=decision,
                outcome="CANCELLED",
                failure_stage=None,
                usage=usage,
            )
            next_input = {
                key: value
                for key, value in input_state.items()
                if key not in {"code_scheduler_decision", "code_runtime_session_id"}
            }
            next_input["code_parent_attempt_id"] = old_attempt_id
            next_input["code_superseded_attempt_records"] = [
                *session.superseded_attempt_records,
                terminal["code_attempt_final_record"],
            ]
            return await self._run(
                next_input,
                thread_id=thread_id,
                pause_control=pause_control,
            )

        try:
            session.pair = self.sandbox_manager.handoff(session.pair, "WORKER")
            self._commit_session_checkpoint(
                session,
                phase="WORKER_RUNNING",
                reason="Scheduler CONTINUE returned ownership to Worker.",
            )
            worker_details = await self._invoke(
                session.worker,
                render_prompt(
                    "runtime/code_scheduler_continue_handoff",
                    scheduler_epoch=session.loop.scheduler_epoch,
                ),
                thread_id=session.worker_thread_id,
                state=usage.state(
                    self._session_state(input_state, session),
                    reserve_model=3,
                    reserve_tools=3,
                ),
            )
            usage.add(worker_details)
            session.all_messages.extend(
                worker_details.get("current_turn_messages", [])
            )
            session.loop = CodeReviewLoopState.model_validate(
                worker_details.get("code_review_loop")
            )
            session.candidate = session.loop.candidate
            updated_submission = worker_details.get("code_worker_submission")
            if isinstance(updated_submission, dict):
                session.worker_submission = updated_submission
            if session.loop.status != "REVIEWING":
                from workers.code_finalization import missing_handoff_message
                raise RuntimeError(missing_handoff_message(worker_details, "SCHEDULER_CONTINUE"))

            review_details: dict[str, Any] = {}
            for review_pass in range(session.loop.max_repair_rounds + 1):
                session.pair = self.sandbox_manager.handoff(
                    session.pair,
                    "REVIEWER",
                )
                session.latest_candidate_relative = (
                    f"candidate/revision-{session.candidate.candidate_revision}-"
                    f"epoch-{session.loop.scheduler_epoch}-review-{review_pass + 1}"
                )
                candidate_snapshot = (
                    session.attempt_root / session.latest_candidate_relative
                )
                self.sandbox_manager.export_candidate(
                    session.pair,
                    candidate_snapshot,
                )
                if pause_control is not None and pause_control.requested:
                    session.pair = self.sandbox_manager.freeze(session.pair)
                    self._commit_session_checkpoint(
                        session,
                        phase="CANDIDATE_READY",
                        reason=(
                            "INSERT paused continued Code after a complete "
                            "Worker turn."
                        ),
                    )
                    await pause_control.pause_point(
                        on_pause=self._writer_slots.release,
                        on_resume=self._writer_slots.acquire,
                    )
                    session.pair = self.sandbox_manager.handoff(
                        session.pair,
                        "REVIEWER",
                    )
                self._commit_session_checkpoint(
                    session,
                    phase="REVIEWER_RUNNING",
                    reason=(
                        "Reviewer role turn is non-preemptible; external control "
                        "is deferred until this invocation returns."
                    ),
                )
                reviewer_state = {
                    "code_tool_audit": _review_tool_audit({"messages": session.all_messages})[0],
                    **self._session_state(input_state, session),
                    "code_publication_context": CodePublicationContext(
                        candidate_root=str(candidate_snapshot),
                        target_root=str(session.integration_root),
                        base_revision=session.base_revision,
                    ).model_dump(mode="json"),
                }
                review_details = await self._invoke(
                    session.reviewer,
                    render_prompt(
                        "runtime/code_review_handoff",
                        review_pass=review_pass + 1,
                    ),
                    thread_id=session.reviewer_thread_id,
                    state=usage.state(reviewer_state),
                )
                usage.add(review_details)
                session.all_messages.extend(
                    review_details.get("current_turn_messages", [])
                )
                session.loop = CodeReviewLoopState.model_validate(
                    review_details.get("code_review_loop")
                )
                session.candidate = session.loop.candidate
                boundary_report = CodeReviewReport.model_validate(
                    review_details.get("code_review_report")
                )
                session.last_review_details = review_details
                session.last_report = boundary_report
                self._commit_session_checkpoint(
                    session,
                    phase="REVIEW_COMPLETED",
                    reason="Reviewer completed its indivisible role turn.",
                )
                if session.loop.status in {
                    "APPLIED",
                    "ESCALATED_TO_SCHEDULER",
                }:
                    break
                if session.loop.status != "WAITING_FOR_WORKER":
                    raise RuntimeError(
                        "Code Reviewer returned an invalid continued loop state: "
                        f"{session.loop.status}"
                    )
                session.pair = self.sandbox_manager.handoff(
                    session.pair,
                    "WORKER",
                )
                self._commit_session_checkpoint(
                    session,
                    phase="WORKER_RUNNING",
                    reason=(
                        "Reviewer handed one complete repair instruction to Worker."
                    ),
                )
                repair_state = {
                    **self._session_state(input_state, session),
                    "code_repair_instruction": (
                        session.loop.pending_instruction.model_dump(mode="json")
                        if session.loop.pending_instruction is not None
                        else None
                    ),
                }
                worker_details = await self._invoke(
                    session.worker,
                    render_prompt(
                        "runtime/code_repair_handoff",
                        repair_round=session.loop.repair_round,
                    ),
                    thread_id=session.worker_thread_id,
                    state=usage.state(
                        repair_state,
                        reserve_model=2,
                        reserve_tools=2,
                    ),
                )
                usage.add(worker_details)
                session.all_messages.extend(
                    worker_details.get("current_turn_messages", [])
                )
                session.loop = CodeReviewLoopState.model_validate(
                    worker_details.get("code_review_loop")
                )
                if session.loop.status != "REVIEWING":
                    from workers.code_finalization import missing_handoff_message
                    raise RuntimeError(missing_handoff_message(worker_details, "CONTINUED_REPAIR_RESPONSE"))
                session.candidate = session.loop.candidate
                updated_submission = worker_details.get("code_worker_submission")
                if isinstance(updated_submission, dict):
                    session.worker_submission = updated_submission
            else:
                raise RuntimeError("Continued Code review exceeded its repair bound")

            report = CodeReviewReport.model_validate(
                review_details.get("code_review_report")
            )
            session.last_review_details = review_details
            session.last_report = report
            if session.loop.status == "APPLIED":
                return self._finish_applied_session(
                    session,
                    report=report,
                    usage=usage,
                    scheduler_decision=decision,
                )
            session.pair = self.sandbox_manager.freeze(session.pair)
            self._commit_session_checkpoint(
                session,
                phase="AWAITING_SCHEDULER",
                reason="Frozen pair is waiting for another Scheduler decision.",
            )
            self._sessions[session.session_id] = session
            return self._paused_result(
                session,
                report=report,
                usage=usage,
                scheduler_decision=decision,
            )
        except Exception as error:
            try:
                session.pair = self.sandbox_manager.freeze(session.pair)
                self._commit_session_checkpoint(
                    session,
                    phase="RECOVERY_REQUIRED",
                    reason=(
                        "Continued runtime stopped before cleanup authorization: "
                        f"{type(error).__name__}"
                    ),
                )
            finally:
                # The durable record remains the owner.  Recovery, not this
                # failing stack frame, decides whether the pair can be reused
                # or safely removed.
                self._sessions.pop(session.session_id, None)
            raise


__all__ = ["CodeStepRuntime"]
