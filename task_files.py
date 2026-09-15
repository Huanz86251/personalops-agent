"""Host-owned private file registration, never a model-callable path importer."""

from datetime import datetime, timezone
import hashlib
import mimetypes
import os
from pathlib import Path
from uuid import uuid4

from artifact_models import DownloadedArtifactRecord
from file_limits import TASK_FILE_MAX_BYTES
from path import AGENT_DATA_ROOT

TASK_FILE_ROOT = AGENT_DATA_ROOT / "task-files"


def identity(state):
    run = str(state.get("event_id") or state.get("planning_run_id") or "").strip()
    worker = str(state.get("worker_id") or "").strip()
    if not run or not worker:
        raise ValueError("Private file access requires run and Worker identity")
    return run, worker


def owner_root(run, worker):
    def key(value):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    return TASK_FILE_ROOT / key(run) / key(worker)


def register_file(source, *, source_root, state, tool_call_id, origin, source_ref):
    """source_root is trusted adapter configuration, never tool arguments."""
    run, worker = identity(state)
    root = Path(source_root).resolve()
    source = Path(source).resolve(strict=True)
    source.relative_to(root)
    if not source.is_file() or source.stat().st_size > TASK_FILE_MAX_BYTES:
        raise ValueError("Downloaded file missing or exceeds 20 MiB")
    with source.open("rb") as stream:
        data = stream.read(TASK_FILE_MAX_BYTES + 1)
    if len(data) > TASK_FILE_MAX_BYTES:
        raise ValueError("Downloaded file exceeds 20 MiB")
    sha = hashlib.sha256(data).hexdigest()
    key = hashlib.sha256((origin + source_ref + sha).encode("utf-8")).hexdigest()[:24]
    candidate_id = "file-" + key
    name = source.name[:200]
    directory = owner_root(run, worker) / candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise ValueError("Registered private copy was modified; refusing reuse")
    else:
        temporary = directory / (uuid4().hex + ".part")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return DownloadedArtifactRecord(
        candidate_id=candidate_id,
        source_url=source_ref,
        storage_path=str(target.resolve()),
        filename=name,
        size_bytes=len(data),
        sha256=sha,
        media_type=mimetypes.guess_type(name)[0],
        tool_call_id=tool_call_id,
        downloaded_at=datetime.now(timezone.utc),
        run_id=run,
        worker_id=worker,
        origin=origin,
    )


def read_download(path, state):
    key = path.removeprefix("/downloads/")
    record = next(
        (
            r
            for r in reversed(state.get("worker_downloaded_artifacts") or [])
            if r.get("candidate_id") == key
        ),
        None,
    )
    if not record:
        raise ValueError("Download is not registered to this Worker")
    record = DownloadedArtifactRecord.model_validate(record)
    if record.run_id is not None:
        run, worker = identity(state)
        if (record.run_id, record.worker_id) != (run, worker):
            raise ValueError("Download belongs to a different run or Worker")
        root = owner_root(run, worker).resolve()
    else:
        from tools.web_artifacts import WEB_ARTIFACT_ROOT, _owner_directory

        worker = str(state.get("worker_id") or "")
        if not worker:
            raise ValueError("Missing Worker identity")
        root = _owner_directory(WEB_ARTIFACT_ROOT, worker).resolve()
    actual = Path(record.storage_path).resolve(strict=True)
    actual.relative_to(root)
    if actual.stat().st_size > TASK_FILE_MAX_BYTES:
        raise ValueError("File exceeds 20 MiB")
    with actual.open("rb") as stream:
        data = stream.read(TASK_FILE_MAX_BYTES + 1)
    if (
        len(data) != record.size_bytes
        or hashlib.sha256(data).hexdigest() != record.sha256
    ):
        raise ValueError("Download content changed since registration")
    return data
