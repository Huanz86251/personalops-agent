"""Tests for run-scoped workspace boundaries and deterministic publication."""

import json
from pathlib import Path

from git import Actor, Repo

from artifact_models import ResolvedArtifactCandidate
from artifact_publisher import publish_artifact_to_handoff
from planning_models import StepArtifactOutput
from run_workspace import (
    inherit_replacement_workspace,
    initialize_code_integration,
    initialize_run_workspace,
    read_replacement_inheritance_receipt,
    write_run_cancellation_receipt,
)
from integration_repository import IntegrationRepository


def test_run_layout_separates_handoff_and_canonical_workspace(
    tmp_path: Path,
) -> None:
    layout = initialize_run_workspace(
        "planning-run-1",
        storage_root=tmp_path / "agent-runs",
        canonical_root=tmp_path / "workspace",
    )

    assert layout.handoff_root.is_dir()
    assert layout.integration_root.is_dir()
    assert layout.worker_private_root("worker-a") != layout.worker_private_root(
        "worker-b"
    )
    assert not layout.canonical_root.is_relative_to(layout.run_root)


def test_harness_publisher_is_content_verified_and_idempotent(
    tmp_path: Path,
) -> None:
    layout = initialize_run_workspace(
        "planning-run-2",
        storage_root=tmp_path / "agent-runs",
        canonical_root=tmp_path / "workspace",
    )
    source = tmp_path / "private" / "copy.md"
    source.parent.mkdir(parents=True)
    source.write_text("approved copy", encoding="utf-8")

    import hashlib

    payload = source.read_bytes()
    candidate = ResolvedArtifactCandidate(
        candidate_id="web-copy-1",
        kind="DOWNLOADED_FILE",
        description="Approved advertisement copy",
        verified=True,
        location=str(source),
        storage_path=str(source),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    first = publish_artifact_to_handoff(layout=layout, candidate=candidate)
    second = publish_artifact_to_handoff(layout=layout, candidate=candidate)

    assert first == second
    assert first.handoff_path.startswith("/handoff/artifact-")
    assert Path(first.storage_path).read_bytes() == payload
    assert not list(layout.canonical_root.iterdir())


def test_remote_repository_is_published_as_reference_metadata(
    tmp_path: Path,
) -> None:
    layout = initialize_run_workspace(
        "planning-run-3",
        storage_root=tmp_path / "agent-runs",
        canonical_root=tmp_path / "workspace",
    )
    candidate = ResolvedArtifactCandidate(
        candidate_id="repo-1",
        kind="REMOTE_REPOSITORY",
        description="Reference repository",
        verified=True,
        location="https://github.com/langchain-ai/deepagents@main",
        source_url="https://github.com/langchain-ai/deepagents",
        repository_ref="main",
        relevant_paths=["libs/deepagents"],
    )

    receipt = publish_artifact_to_handoff(
        layout=layout,
        candidate=candidate,
    )

    content = Path(receipt.storage_path).read_text(encoding="utf-8")
    assert "langchain-ai/deepagents" in content
    assert receipt.handoff_path.endswith("repository-reference.json")


def test_user_deliverable_intent_is_persisted_in_handoff_receipt(
    tmp_path: Path,
) -> None:
    layout = initialize_run_workspace(
        "planning-run-deliverable",
        storage_root=tmp_path / "agent-runs",
        canonical_root=tmp_path / "workspace",
    )
    source = tmp_path / "downloaded.html"
    source.write_text("<main>approved</main>", encoding="utf-8")
    payload = source.read_bytes()
    import hashlib

    candidate = ResolvedArtifactCandidate(
        candidate_id="downloaded-html",
        output_id="requested_html",
        kind="DOWNLOADED_FILE",
        description="Requested HTML",
        verified=True,
        location=str(source),
        storage_path=str(source),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    receipt = publish_artifact_to_handoff(
        layout=layout,
        candidate=candidate,
        output=StepArtifactOutput(
            output_id="requested_html",
            description="Requested HTML",
            disposition="USER_DELIVERABLE",
            target_path="downloads/template.html",
        ),
    )

    assert receipt.disposition == "USER_DELIVERABLE"
    assert receipt.target_path == "downloads/template.html"


def test_code_integration_is_seeded_once_and_preserves_run_progress(
    tmp_path: Path,
) -> None:
    source = tmp_path / "workspace"
    source.mkdir()
    (source / "app.py").write_text("base", encoding="utf-8")
    layout = initialize_run_workspace(
        "planning-run-code",
        storage_root=tmp_path / "agent-runs",
        canonical_root=source,
    )

    integration = initialize_code_integration(
        layout=layout,
        source_root=source,
    )
    (integration / "app.py").write_text("step-one", encoding="utf-8")
    (source / "app.py").write_text("external-change", encoding="utf-8")

    resumed = initialize_code_integration(
        layout=layout,
        source_root=source,
    )

    assert resumed == integration
    assert (resumed / "app.py").read_text(encoding="utf-8") == "step-one"


def test_cancellation_receipt_keeps_accepted_and_private_locations_auditable(
    tmp_path: Path,
) -> None:
    layout = initialize_run_workspace(
        "event-cancelled-run",
        storage_root=tmp_path / "agent-runs",
        canonical_root=tmp_path / "workspace",
    )

    receipt = write_run_cancellation_receipt(
        layout=layout,
        cancel_event_id="event-cancel-command",
        target_event_id="event-cancelled-run",
        code_final_records=({"record_id": "code-final-1"},),
    )
    repeated = write_run_cancellation_receipt(
        layout=layout,
        cancel_event_id="event-cancel-command",
        target_event_id="event-cancelled-run",
        code_final_records=({"record_id": "code-final-1"},),
    )

    assert receipt == repeated
    record = json.loads(receipt.read_text(encoding="utf-8"))
    assert record["kind"] == "RUN_CANCELLED"
    assert record["preserved"]["handoff_root"] == str(layout.handoff_root)
    assert record["preserved"]["integration_root"] == str(
        layout.integration_root
    )
    assert record["preserved"]["code_attempt_record_ids"] == [
        "code-final-1"
    ]


def test_replacement_inherits_only_handoff_and_clean_git_history(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("base", encoding="utf-8")
    old_layout = initialize_run_workspace(
        "old-run",
        storage_root=tmp_path / "runs",
        canonical_root=workspace,
    )
    new_layout = initialize_run_workspace(
        "replacement-run",
        storage_root=tmp_path / "runs",
        canonical_root=workspace,
    )
    (old_layout.handoff_root / "research.md").write_text(
        "approved research",
        encoding="utf-8",
    )
    (old_layout.private_root / "secret-draft.md").write_text(
        "unreviewed",
        encoding="utf-8",
    )
    (old_layout.candidates_root / "candidate.py").write_text(
        "unreviewed",
        encoding="utf-8",
    )
    initialize_code_integration(layout=old_layout, source_root=workspace)
    repository = IntegrationRepository(
        run_id=old_layout.run_id,
        root=old_layout.integration_root,
        receipts_root=old_layout.receipts_root,
    )
    repository.initialize()
    (old_layout.integration_root / "app.py").write_text(
        "accepted",
        encoding="utf-8",
    )
    actor = Actor("Test Harness", "test@example.invalid")
    with Repo(old_layout.integration_root) as repo:
        repo.git.add("-A")
        accepted_head = repo.index.commit(
            "accepted change",
            author=actor,
            committer=actor,
        ).hexsha

    receipt_path = inherit_replacement_workspace(
        source_layout=old_layout,
        replacement_layout=new_layout,
        replacement_event_id=new_layout.run_id,
    )

    receipt = read_replacement_inheritance_receipt(layout=new_layout)
    assert receipt_path.is_file()
    assert receipt["handoff"]["file_count"] == 1
    assert (
        Path(receipt["handoff"]["root"]) / "research.md"
    ).read_text(encoding="utf-8") == "approved research"
    assert not (new_layout.private_root / "secret-draft.md").exists()
    assert not (new_layout.candidates_root / "candidate.py").exists()
    with Repo(new_layout.integration_root) as repo:
        assert repo.head.commit.hexsha == accepted_head
    assert IntegrationRepository(
        run_id=new_layout.run_id,
        root=new_layout.integration_root,
        receipts_root=new_layout.receipts_root,
    ).status().baseline_commit == accepted_head
