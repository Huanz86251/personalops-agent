"""Bind an import destination to the current file tool, independently of Worker factories."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_DESTINATION = ContextVar("task_file_import_destination", default=None)


@contextmanager
def import_destination(sandbox, root):
    token = _DESTINATION.set((sandbox, root))
    try:
        yield
    finally:
        _DESTINATION.reset(token)


def materialize_import(record, sandbox=None, root="/workspace/.private-downloads"):
    if sandbox is None:
        destination = _DESTINATION.get()
        if destination is None or destination[0] is None:
            return record
        sandbox, root = destination
    filename = record.filename
    if not filename or Path(filename).name != filename:
        raise ValueError("Registered import filename is invalid")
    path = f"{root}/{record.candidate_id}/{filename}"
    payload = Path(record.storage_path).read_bytes()
    import hashlib
    if len(payload) != record.size_bytes or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise ValueError("Registered import changed before sandbox copy")
    try:
        uploaded = sandbox.upload_files([(path, payload)])
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Sandbox import failed: {error}") from error
    if len(uploaded) != 1 or uploaded[0].error:
        detail = uploaded[0].error if uploaded else "missing copy receipt"
        raise ValueError(f"Sandbox import failed: {detail}")
    return record.model_copy(update={"sandbox_path": path})
