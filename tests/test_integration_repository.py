"""Tests for the run-local Git integration history."""

from pathlib import Path

import pytest

from integration_repository import IntegrationRepository


def test_baseline_and_accepted_commit_are_real_git_history(tmp_path: Path) -> None:
    root = tmp_path / "integration"
    root.mkdir()
    (root / "app.py").write_text("print('base')\n", encoding="utf-8")
    repository = IntegrationRepository(
        run_id="run-git-history",
        root=root,
        receipts_root=tmp_path / "receipts",
    )

    baseline = repository.initialize()
    initial = repository.status()
    assert initial.baseline_commit == baseline.baseline_commit
    assert initial.head_commit == baseline.baseline_commit
    assert initial.working_tree_clean

    (root / "app.py").write_text("print('accepted')\n", encoding="utf-8")
    receipt = repository.commit_accepted(
        step_id=1,
        candidate_revision=2,
        manifest_id="manifest-1",
        publication_id="publication-1",
        approved_paths=("app.py",),
        review_summary="Reviewer approved the behavior.",
        test_summary="One focused test passed.",
    )

    status = repository.status()
    assert status.head_commit == receipt.accepted_commit
    assert status.head_commit != status.baseline_commit
    assert status.working_tree_clean
    assert status.accepted_commit_count == 1
    assert repository.history(max_entries=2)[0].commit == receipt.accepted_commit
    diff = repository.diff(status.baseline_commit)
    assert "accepted" in diff.text
    assert not diff.truncated
    view = repository.show_commit(receipt.accepted_commit)
    assert view.commit == receipt.accepted_commit
    assert view.changed_files == ("app.py",)
    assert "Reviewer approved the behavior" in view.message


def test_commit_is_idempotent_and_rejects_unapproved_changes(tmp_path: Path) -> None:
    root = tmp_path / "integration"
    root.mkdir()
    (root / "app.py").write_text("base", encoding="utf-8")
    repository = IntegrationRepository(
        run_id="run-idempotent",
        root=root,
        receipts_root=tmp_path / "receipts",
    )
    repository.initialize()
    (root / "app.py").write_text("accepted", encoding="utf-8")
    (root / "unexpected.py").write_text("not approved", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the approved manifest"):
        repository.commit_accepted(
            step_id=1,
            candidate_revision=1,
            manifest_id="manifest-2",
            publication_id="publication-2",
            approved_paths=("app.py",),
            review_summary="Approved app only.",
            test_summary="Test passed.",
        )

    (root / "unexpected.py").unlink()
    first = repository.commit_accepted(
        step_id=1,
        candidate_revision=1,
        manifest_id="manifest-2",
        publication_id="publication-2",
        approved_paths=("app.py",),
        review_summary="Approved app only.",
        test_summary="Test passed.",
    )
    second = repository.commit_accepted(
        step_id=1,
        candidate_revision=1,
        manifest_id="manifest-2",
        publication_id="publication-2",
        approved_paths=("app.py",),
        review_summary="Approved app only.",
        test_summary="Test passed.",
    )
    assert first == second
    assert len(repository.history(max_entries=10)) == 2
