"""Run-scoped storage boundaries for private work, handoff, and integration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Sequence
from uuid import uuid4

from git import Repo

from deepagents import FilesystemPermission
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol

from path import AGENT_DATA_ROOT, WORKSPACE_ROOT


RUN_WORKSPACE_ROOT = AGENT_DATA_ROOT / "runs"
CONVERSATION_WORKSPACE_ROOT = WORKSPACE_ROOT / "conversations"
HANDOFF_VIRTUAL_ROOT = "/handoff"


def stable_storage_key(value: str, *, prefix: str) -> str:
    """Turn an external identity into a Windows-safe stable directory key."""

    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{prefix} identity cannot be empty")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class RunWorkspaceLayout:
    """All storage owned by one Planning run.

    ``canonical_root`` is deliberately outside ``run_root``. Nothing under a
    run becomes a user deliverable merely because a Worker or Reporter ended.
    """

    run_id: str
    run_root: Path
    private_root: Path
    candidates_root: Path
    handoff_root: Path
    integration_root: Path
    staging_root: Path
    receipts_root: Path
    canonical_root: Path

    def worker_private_root(self, worker_id: str) -> Path:
        return self.private_root / stable_storage_key(worker_id, prefix="worker")


def initialize_run_workspace(
    run_id: str,
    *,
    storage_root: Path = RUN_WORKSPACE_ROOT,
    canonical_root: Path = WORKSPACE_ROOT,
) -> RunWorkspaceLayout:
    """Create and return the deterministic directory layout for one run."""

    normalized_run_id = str(run_id).strip()
    run_root = Path(storage_root).resolve() / stable_storage_key(
        normalized_run_id,
        prefix="run",
    )
    layout = RunWorkspaceLayout(
        run_id=normalized_run_id,
        run_root=run_root,
        private_root=run_root / "private",
        candidates_root=run_root / "candidates",
        handoff_root=run_root / "handoff",
        integration_root=run_root / "integration",
        staging_root=run_root / "staging",
        receipts_root=run_root / "receipts",
        canonical_root=Path(canonical_root).resolve(),
    )
    for directory in (
        layout.private_root,
        layout.candidates_root,
        layout.handoff_root,
        layout.integration_root,
        layout.staging_root,
        layout.receipts_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    layout.canonical_root.mkdir(parents=True, exist_ok=True)
    metadata_path = layout.run_root / "run-metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_id") != normalized_run_id:
            raise ValueError("run workspace identity conflict")
    else:
        temporary = layout.staging_root / f".run-metadata.{uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(
                    {
                        "run_id": normalized_run_id,
                        "run_root": str(layout.run_root),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
    return layout


def initialize_conversation_workspace(
    conversation_id: str,
    *,
    root: Path = CONVERSATION_WORKSPACE_ROOT,
) -> Path:
    """Return the stable user-visible workspace owned by one conversation."""

    workspace = Path(root).resolve() / stable_storage_key(
        conversation_id,
        prefix="conversation",
    )
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace.resolve()


def cleanup_expired_run_workspaces(
    *,
    terminal_runs: dict[str, datetime],
    protected_run_ids: set[str],
    retention_minutes: int,
    storage_root: Path = RUN_WORKSPACE_ROOT,
    now: datetime | None = None,
) -> tuple[dict[str, object], ...]:
    """Delete expired bulky run data while retaining receipts and identity."""

    if retention_minutes < 1:
        raise ValueError("retention_minutes must be positive")
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("now must include timezone information")
    cutoff = timestamp.astimezone(timezone.utc) - timedelta(
        minutes=retention_minutes
    )
    root = Path(storage_root).resolve()
    if not root.is_dir():
        return ()
    outcomes: list[dict[str, object]] = []
    for run_root in sorted(root.iterdir()):
        if not run_root.is_dir():
            continue
        metadata_path = run_root / "run-metadata.json"
        if not metadata_path.is_file():
            outcomes.append(
                {"status": "SKIPPED", "root": str(run_root), "reason": "missing metadata"}
            )
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            run_id = str(metadata.get("run_id") or "").strip()
        except Exception:
            outcomes.append(
                {"status": "SKIPPED", "root": str(run_root), "reason": "invalid metadata"}
            )
            continue
        changed_at = terminal_runs.get(run_id)
        if run_id in protected_run_ids or changed_at is None or changed_at > cutoff:
            continue
        removed: list[str] = []
        for name in ("private", "candidates", "handoff", "integration", "staging"):
            target = (run_root / name).resolve()
            if target.parent != run_root.resolve():
                raise RuntimeError("run cleanup target escaped run root")
            if target.is_dir():
                shutil.rmtree(target)
                removed.append(name)
        receipt_root = run_root / "receipts"
        receipt_root.mkdir(parents=True, exist_ok=True)
        cleanup_receipt = receipt_root / "retention-cleanup.json"
        if not cleanup_receipt.is_file():
            cleanup_receipt.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "cleaned_at": timestamp.astimezone(timezone.utc).isoformat().replace(
                            "+00:00", "Z"
                        ),
                        "removed_directories": removed,
                        "policy": (
                            "Bulky run-owned data expired; compact receipts and "
                            "the conversation workspace were retained."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        outcomes.append(
            {"status": "CLEANED", "run_id": run_id, "removed": tuple(removed)}
        )
    return tuple(outcomes)


def write_run_cancellation_receipt(
    *,
    layout: RunWorkspaceLayout,
    cancel_event_id: str,
    target_event_id: str,
    code_final_records: Sequence[dict] = (),
) -> Path:
    """Atomically record what survived after a run-level cancellation."""

    normalized_cancel_id = str(cancel_event_id).strip()
    normalized_target_id = str(target_event_id).strip()
    if not normalized_cancel_id or not normalized_target_id:
        raise ValueError("Cancellation receipt requires both Event identities")
    if normalized_target_id != layout.run_id:
        raise ValueError("Cancellation receipt does not belong to this run")

    receipt_name = stable_storage_key(
        normalized_cancel_id,
        prefix="cancel",
    ) + ".json"
    destination = layout.receipts_root / receipt_name
    record = {
        "kind": "RUN_CANCELLED",
        "cancel_event_id": normalized_cancel_id,
        "target_event_id": normalized_target_id,
        "cancelled_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "preserved": {
            "run_root": str(layout.run_root),
            "private_root": str(layout.private_root),
            "candidates_root": str(layout.candidates_root),
            "handoff_root": str(layout.handoff_root),
            "integration_root": str(layout.integration_root),
            "planning_checkpoint": "SQLite checkpoint retained by thread_id",
            "code_attempt_record_ids": [
                str(item.get("record_id") or "")
                for item in code_final_records
                if str(item.get("record_id") or "").strip()
            ],
        },
        "publication_policy": (
            "Already accepted handoff/integration artifacts are retained; "
            "unreviewed cancellation snapshots are archive-only."
        ),
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if (
            existing.get("cancel_event_id") != normalized_cancel_id
            or existing.get("target_event_id") != normalized_target_id
        ):
            raise ValueError("Cancellation receipt identity conflict")
        return destination.resolve()

    temporary = layout.staging_root / f".{receipt_name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def write_run_supersession_receipt(
    *,
    layout: RunWorkspaceLayout,
    replacement_event_id: str,
    target_event_id: str,
    code_final_records: Sequence[dict] = (),
) -> Path:
    """Atomically record the validated state available to a replacement run."""

    normalized_replacement_id = str(replacement_event_id).strip()
    normalized_target_id = str(target_event_id).strip()
    if not normalized_replacement_id or not normalized_target_id:
        raise ValueError("Supersession receipt requires both Event identities")
    if normalized_target_id != layout.run_id:
        raise ValueError("Supersession receipt does not belong to this run")

    receipt_name = stable_storage_key(
        normalized_replacement_id,
        prefix="replace",
    ) + ".json"
    destination = layout.receipts_root / receipt_name
    record = {
        "kind": "RUN_SUPERSEDED",
        "replacement_event_id": normalized_replacement_id,
        "target_event_id": normalized_target_id,
        "superseded_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "preserved": {
            "run_root": str(layout.run_root),
            "handoff_root": str(layout.handoff_root),
            "integration_root": str(layout.integration_root),
            "planning_checkpoint": "SQLite checkpoint retained by thread_id",
            "code_attempt_record_ids": [
                str(item.get("record_id") or "")
                for item in code_final_records
                if str(item.get("record_id") or "").strip()
            ],
        },
        "inheritance_policy": (
            "The replacement planner may reuse accepted StepReports, handoff "
            "artifacts, and committed integration history. Unreviewed private "
            "or candidate files remain archive-only."
        ),
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if (
            existing.get("replacement_event_id") != normalized_replacement_id
            or existing.get("target_event_id") != normalized_target_id
        ):
            raise ValueError("Supersession receipt identity conflict")
        return destination.resolve()

    temporary = layout.staging_root / f".{receipt_name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def read_run_supersession_receipt(
    *,
    layout: RunWorkspaceLayout,
    replacement_event_id: str,
) -> dict | None:
    """Read the idempotent supersession receipt for one replacement Event."""

    normalized_replacement_id = str(replacement_event_id).strip()
    if not normalized_replacement_id:
        raise ValueError("replacement_event_id cannot be empty")
    receipt_name = stable_storage_key(
        normalized_replacement_id,
        prefix="replace",
    ) + ".json"
    path = layout.receipts_root / receipt_name
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("replacement_event_id") != normalized_replacement_id
        or record.get("target_event_id") != layout.run_id
    ):
        raise ValueError("Supersession receipt identity conflict")
    return record


def inherit_replacement_workspace(
    *,
    source_layout: RunWorkspaceLayout,
    replacement_layout: RunWorkspaceLayout,
    replacement_event_id: str,
) -> Path:
    """Copy only accepted handoff and committed Git state into a new run."""

    normalized_replacement_id = str(replacement_event_id).strip()
    if not normalized_replacement_id:
        raise ValueError("replacement_event_id cannot be empty")
    if normalized_replacement_id != replacement_layout.run_id:
        raise ValueError("Replacement layout does not match replacement Event")
    receipt = replacement_layout.receipts_root / "replacement-inheritance.json"
    if receipt.is_file():
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            record.get("source_run_id") != source_layout.run_id
            or record.get("replacement_event_id") != normalized_replacement_id
        ):
            raise ValueError("Replacement inheritance identity conflict")
        return receipt.resolve()

    inherited_handoff = (
        replacement_layout.handoff_root
        / "inherited"
        / stable_storage_key(source_layout.run_id, prefix="run")
    )
    handoff_file_count = 0
    if source_layout.handoff_root.is_dir():
        for source in source_layout.handoff_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(source_layout.handoff_root)
            destination = inherited_handoff / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            handoff_file_count += 1

    integration_record: dict[str, object] = {
        "inherited": False,
        "reason": "The superseded run has no committed integration repository.",
    }
    source_git = source_layout.integration_root / ".git"
    replacement_marker = replacement_layout.run_root / "integration-seed.json"
    if source_git.is_dir():
        with Repo(source_layout.integration_root) as source_repo:
            changed = bool(source_repo.is_dirty(untracked_files=True))
            source_head = source_repo.head.commit.hexsha
        if changed:
            integration_record = {
                "inherited": False,
                "reason": (
                    "The superseded integration working tree was dirty; only "
                    "committed and clean integration state may be inherited."
                ),
                "source_head": source_head,
            }
        else:
            transaction = (
                replacement_layout.staging_root
                / f"replacement-integration-{uuid4().hex}"
            )
            staged_tree = transaction / "tree"
            transaction.mkdir(parents=True, exist_ok=False)
            try:
                shutil.copytree(source_layout.integration_root, staged_tree)
                if any(replacement_layout.integration_root.iterdir()):
                    raise ValueError(
                        "replacement integration contains unowned data"
                    )
                replacement_layout.integration_root.rmdir()
                os.replace(staged_tree, replacement_layout.integration_root)
                baseline = {
                    "run_id": replacement_layout.run_id,
                    "repository_root": str(replacement_layout.integration_root),
                    "baseline_commit": source_head,
                    "created_at": datetime.now(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    ),
                }
                baseline_path = (
                    replacement_layout.receipts_root
                    / "integration-baseline.json"
                )
                baseline_path.write_text(
                    json.dumps(baseline, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                replacement_marker.write_text(
                    json.dumps(
                        {
                            "source_root": str(source_layout.integration_root),
                            "integration_root": str(
                                replacement_layout.integration_root
                            ),
                            "inherited_from_run_id": source_layout.run_id,
                            "inherited_head": source_head,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            finally:
                shutil.rmtree(transaction, ignore_errors=True)
            integration_record = {
                "inherited": True,
                "source_head": source_head,
                "replacement_baseline": source_head,
                "repository_root": str(replacement_layout.integration_root),
            }

    record = {
        "kind": "REPLACEMENT_INHERITANCE",
        "source_run_id": source_layout.run_id,
        "replacement_event_id": normalized_replacement_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "handoff": {
            "file_count": handoff_file_count,
            "root": str(inherited_handoff),
        },
        "integration": integration_record,
        "policy": (
            "Only Reporter-published handoff files and a clean committed Git "
            "integration tree are inherited. Private and candidate roots are not."
        ),
    }
    temporary = replacement_layout.staging_root / (
        f".replacement-inheritance.{uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt.resolve()


def read_replacement_inheritance_receipt(
    *,
    layout: RunWorkspaceLayout,
) -> dict | None:
    path = layout.receipts_root / "replacement-inheritance.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("replacement_event_id") != layout.run_id:
        raise ValueError("Replacement inheritance receipt identity conflict")
    return record


def materialize_worker_artifact(
    *,
    layout: RunWorkspaceLayout,
    worker_id: str,
    candidate_id: str,
    filename: str,
    payload: bytes,
) -> Path:
    """Freeze one state-backed Worker file into Harness-owned storage."""

    safe_name = Path(str(filename).replace("\\", "/")).name.strip()
    safe_name = "".join(
        character
        if character.isalnum() or character in {".", "_", "-"}
        else "_"
        for character in safe_name
    ).strip("._") or "artifact.bin"
    destination = (
        layout.candidates_root
        / stable_storage_key(worker_id, prefix="worker")
        / stable_storage_key(candidate_id, prefix="candidate")
        / safe_name[:180]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != payload:
            raise ValueError(
                "artifact candidate identity was reused for different content"
            )
        return destination.resolve()

    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def initialize_code_integration(
    *,
    layout: RunWorkspaceLayout,
    source_root: Path,
) -> Path:
    """Seed one run-local code tree without mutating the user workspace."""

    source = Path(source_root).resolve()
    if not source.is_dir():
        raise ValueError(f"code integration source is not a directory: {source}")
    marker = layout.run_root / "integration-seed.json"
    if marker.is_file():
        return layout.integration_root.resolve()
    if any(layout.integration_root.iterdir()):
        raise ValueError(
            "integration workspace contains data without an initialization marker"
        )

    transaction = layout.staging_root / f"integration-seed-{uuid4().hex}"
    staged_tree = transaction / "tree"
    transaction.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(
            source,
            staged_tree,
            ignore=shutil.ignore_patterns(
                ".agent",
                ".agents",
                ".codex",
                ".git",
                ".pytest_cache",
                ".venv",
                "__pycache__",
                "node_modules",
            ),
        )
        layout.integration_root.rmdir()
        os.replace(staged_tree, layout.integration_root)
        marker_tmp = transaction / "integration-seed.json"
        marker_tmp.write_text(
            json.dumps(
                {"source_root": str(source), "integration_root": str(layout.integration_root)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(marker_tmp, marker)
    finally:
        shutil.rmtree(transaction, ignore_errors=True)
    return layout.integration_root.resolve()


def create_run_worker_backend(
    layout: RunWorkspaceLayout,
    *,
    private_backend: BackendProtocol | None = None,
) -> CompositeBackend:
    """Expose private storage plus the run's shared handoff namespace."""

    return CompositeBackend(
        default=private_backend or StateBackend(),
        routes={
            f"{HANDOFF_VIRTUAL_ROOT}/": FilesystemBackend(
                root_dir=layout.handoff_root,
                virtual_mode=True,
                max_file_size_mb=100,
            )
        },
        artifacts_root="/private/artifacts",
    )


def handoff_read_only_permissions() -> Sequence[FilesystemPermission]:
    """Deny all model-authored mutations under the shared handoff route."""

    return (
        FilesystemPermission(
            operations=["write"],
            paths=[f"{HANDOFF_VIRTUAL_ROOT}/**"],
            mode="deny",
        ),
    )


__all__ = [
    "CONVERSATION_WORKSPACE_ROOT",
    "HANDOFF_VIRTUAL_ROOT",
    "RUN_WORKSPACE_ROOT",
    "RunWorkspaceLayout",
    "create_run_worker_backend",
    "cleanup_expired_run_workspaces",
    "handoff_read_only_permissions",
    "initialize_code_integration",
    "initialize_conversation_workspace",
    "initialize_run_workspace",
    "inherit_replacement_workspace",
    "materialize_worker_artifact",
    "read_run_supersession_receipt",
    "read_replacement_inheritance_receipt",
    "stable_storage_key",
    "write_run_cancellation_receipt",
    "write_run_supersession_receipt",
]
