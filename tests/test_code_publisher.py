"""Offline tests for deterministic Reviewer-authorized publication."""

from __future__ import annotations

from pathlib import Path

import pytest

from workers.code_publisher import (
    CodePublicationContext,
    compute_code_tree_revision,
    publish_code_artifacts,
)
from workers.code_review_models import CodeCandidateRef


def candidate() -> CodeCandidateRef:
    return CodeCandidateRef(
        event_id="event-publish-1",
        step_id=1,
        attempt_id="attempt-publish-1",
        workspace_id="workspace-publish-1",
        candidate_revision=3,
    )


def test_publisher_copies_only_reviewer_approved_files(tmp_path: Path) -> None:
    source = tmp_path / "candidate"
    target = tmp_path / "official"
    (source / "src").mkdir(parents=True)
    target.mkdir()
    (source / "src" / "app.py").write_text("new = True\n", encoding="utf-8")
    (source / "debug.log").write_text("not delivered\n", encoding="utf-8")
    (target / "src").mkdir()
    (target / "src" / "app.py").write_text("new = False\n", encoding="utf-8")
    base = compute_code_tree_revision(target)

    manifest, receipt = publish_code_artifacts(
        candidate=candidate(),
        approved_artifact_paths=("src/app.py",),
        context=CodePublicationContext(
            candidate_root=str(source),
            target_root=str(target),
            base_revision=base,
        ),
    )

    assert (target / "src" / "app.py").read_text(encoding="utf-8") == "new = True\n"
    assert not (target / "debug.log").exists()
    assert tuple(item.path for item in manifest.files) == ("src/app.py",)
    assert receipt.manifest_id == manifest.manifest_id
    assert receipt.applied_revision == compute_code_tree_revision(target)


def test_publisher_rejects_a_changed_official_base(tmp_path: Path) -> None:
    source = tmp_path / "candidate"
    target = tmp_path / "official"
    source.mkdir()
    target.mkdir()
    (source / "app.py").write_text("candidate\n", encoding="utf-8")
    old_base = compute_code_tree_revision(target)
    (target / "other.py").write_text("concurrent change\n", encoding="utf-8")

    with pytest.raises(ValueError, match="base revision changed"):
        publish_code_artifacts(
            candidate=candidate(),
            approved_artifact_paths=("app.py",),
            context=CodePublicationContext(
                candidate_root=str(source),
                target_root=str(target),
                base_revision=old_base,
            ),
        )

    assert not (target / "app.py").exists()


def test_publisher_rolls_back_files_when_one_target_is_invalid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate"
    target = tmp_path / "official"
    source.mkdir()
    target.mkdir()
    (source / "a.txt").write_text("new-a\n", encoding="utf-8")
    (source / "b.txt").write_text("new-b\n", encoding="utf-8")
    (target / "a.txt").write_text("old-a\n", encoding="utf-8")
    (target / "b.txt").mkdir()
    base = compute_code_tree_revision(target)

    with pytest.raises(ValueError, match="not a regular file"):
        publish_code_artifacts(
            candidate=candidate(),
            approved_artifact_paths=("a.txt", "b.txt"),
            context=CodePublicationContext(
                candidate_root=str(source),
                target_root=str(target),
                base_revision=base,
            ),
        )

    assert (target / "a.txt").read_text(encoding="utf-8") == "old-a\n"
    assert (target / "b.txt").is_dir()


@pytest.mark.parametrize("unsafe", ["../escape.py", "/absolute.py", ".git/config"])
def test_publisher_rejects_unsafe_artifact_paths(
    tmp_path: Path,
    unsafe: str,
) -> None:
    source = tmp_path / "candidate"
    target = tmp_path / "official"
    source.mkdir()
    target.mkdir()

    with pytest.raises(ValueError, match="unsafe artifact path"):
        publish_code_artifacts(
            candidate=candidate(),
            approved_artifact_paths=(unsafe,),
            context=CodePublicationContext(
                candidate_root=str(source),
                target_root=str(target),
                base_revision=compute_code_tree_revision(target),
            ),
        )
