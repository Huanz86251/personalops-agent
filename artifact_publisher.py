"""Deterministic publication from reviewed candidates into run handoff."""

from __future__ import annotations
from runtime_tracing import operation

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from artifact_models import ResolvedArtifactCandidate
from planning_models import StepArtifactOutput
from run_workspace import HANDOFF_VIRTUAL_ROOT, RunWorkspaceLayout, stable_storage_key


class ArtifactPublicationReceipt(BaseModel):
    """Harness-authored proof of one immutable handoff publication."""

    publication_id: str
    idempotency_key: str
    run_id: str
    candidate_id: str
    review_ref: str | None = None
    output_id: str | None = None
    disposition: Literal["INTERNAL_HANDOFF", "USER_DELIVERABLE"] = (
        "INTERNAL_HANDOFF"
    )
    target_path: str | None = None
    description: str = ""
    kind: str
    handoff_path: str
    storage_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    source_url: str | None = None
    published_at: datetime


def _safe_filename(value: str) -> str:
    raw = Path(str(value).replace("\\", "/")).name.strip()
    allowed = "".join(
        character
        if character.isalnum() or character in {".", "_", "-"}
        else "_"
        for character in raw
    )
    return (allowed.strip("._") or "artifact.bin")[:180]


def _source_payload(candidate: ResolvedArtifactCandidate) -> tuple[bytes, str]:
    if candidate.kind == "REMOTE_REPOSITORY":
        payload = json.dumps(
            {
                "kind": candidate.kind,
                "repository_url": candidate.source_url,
                "repository_ref": candidate.repository_ref,
                "commit": candidate.commit,
                "relevant_paths": candidate.relevant_paths,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        return payload, "repository-reference.json"

    source_value = str(candidate.storage_path or "").strip()
    if not source_value:
        raise ValueError(
            f"candidate has no Harness storage path: {candidate.candidate_id}"
        )
    source = Path(source_value).resolve()
    if not source.is_file():
        raise ValueError(
            f"candidate source is unavailable: {candidate.candidate_id}"
        )
    payload = source.read_bytes()
    return payload, _safe_filename(source.name)


def _receipt_path(layout: RunWorkspaceLayout, publication_id: str) -> Path:
    return layout.receipts_root / f"{publication_id}.json"


@operation('Artifact Publisher / Publish Handoff', fields=())
def publish_artifact_to_handoff(
    *,
    layout: RunWorkspaceLayout,
    candidate: ResolvedArtifactCandidate,
    output: StepArtifactOutput | None = None,
) -> ArtifactPublicationReceipt:
    """Publish one Reporter-approved candidate without model filesystem access."""

    if not candidate.verified:
        raise ValueError("unverified artifact candidate cannot be published")
    if output is not None and candidate.output_id != output.output_id:
        raise ValueError("artifact candidate is bound to another output contract")
    if output is None and candidate.output_id:
        raise ValueError("artifact candidate references an unknown output contract")
    candidate_identity = candidate.review_ref or candidate.candidate_id
    candidate_key = stable_storage_key(candidate_identity, prefix="artifact")
    disposition = output.disposition if output is not None else "INTERNAL_HANDOFF"
    target_path = output.target_path if output is not None else None
    publication_identity = (
        f"{layout.run_id}:{candidate_identity}:"
        f"{candidate.sha256 or candidate.location}:"
        f"{candidate.output_id or '-'}:{disposition}:{target_path or '-'}"
    )
    publication_hash = hashlib.sha256(
        publication_identity.encode("utf-8")
    ).hexdigest()
    publication_id = f"handoff-{publication_hash[:24]}"
    receipt_path = _receipt_path(layout, publication_id)
    if receipt_path.is_file():
        return ArtifactPublicationReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )

    payload, filename = _source_payload(candidate)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if candidate.sha256 and actual_sha256 != candidate.sha256:
        raise ValueError(
            f"candidate content changed before publication: {candidate.candidate_id}"
        )
    if candidate.size_bytes is not None and len(payload) != candidate.size_bytes:
        raise ValueError(
            f"candidate size changed before publication: {candidate.candidate_id}"
        )

    artifact_root = layout.handoff_root / candidate_key
    destination = artifact_root / filename
    staging = layout.staging_root / f"publication-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    lock_path = layout.run_root / f".{candidate_key}.publish.lock"
    lock_fd: int | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        staged_file = staging / filename
        staged_file.write_bytes(payload)
        if hashlib.sha256(staged_file.read_bytes()).hexdigest() != actual_sha256:
            raise OSError("staged artifact checksum mismatch")
        artifact_root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != actual_sha256:
                raise ValueError("handoff path already contains different content")
        else:
            os.replace(staged_file, destination)

        handoff_path = f"{HANDOFF_VIRTUAL_ROOT}/{candidate_key}/{filename}"
        receipt = ArtifactPublicationReceipt(
            publication_id=publication_id,
            idempotency_key=publication_identity,
            run_id=layout.run_id,
            candidate_id=candidate.candidate_id,
            review_ref=candidate.review_ref,
            output_id=candidate.output_id,
            disposition=disposition,
            target_path=target_path,
            description=candidate.description,
            kind=candidate.kind,
            handoff_path=handoff_path,
            storage_path=str(destination.resolve()),
            sha256=actual_sha256,
            size_bytes=len(payload),
            source_url=candidate.source_url,
            published_at=datetime.now(timezone.utc),
        )
        receipt_tmp = staging / f"{publication_id}.json"
        receipt_tmp.write_text(
            receipt.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(receipt_tmp, receipt_path)
        return receipt
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "ArtifactPublicationReceipt",
    "publish_artifact_to_handoff",
]
