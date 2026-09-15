"""Bounded, streamed downloads owned by an individual Web Worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from artifact_models import (
    DownloadedArtifactRecord,
    WebDownloadFailure,
    WebDownloadToolResult,
    WorkerArtifactCandidate,
)
from path import AGENT_DATA_ROOT
from prompt_loader import load_prompt
from file_limits import TASK_FILE_MAX_BYTES, TASK_FILE_MAX_MIB


DOWNLOAD_WEB_ARTIFACT_NAME = "download_web_artifact"
WEB_ARTIFACT_ROOT = AGENT_DATA_ROOT / "web_worker_artifacts"
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


class WebDownloadError(RuntimeError):
    """A download that the runtime safely rejected."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        limit_bytes: int | None = None,
        announced_bytes: int | None = None,
        observed_bytes: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure = WebDownloadFailure(
            code=code,
            stage=stage,
            message=message,
            limit_bytes=limit_bytes,
            announced_bytes=announced_bytes,
            observed_bytes=observed_bytes,
            retryable=retryable,
        )


def _safe_filename(url: str, requested: str | None) -> str:
    raw_name = str(requested or "").strip()
    if not raw_name:
        raw_name = unquote(Path(urlparse(url).path).name)
    raw_name = Path(raw_name or "download.bin").name
    normalized = _SAFE_FILENAME.sub("_", raw_name).strip("._")
    return (normalized or "download.bin")[:180]


def _validate_url(url: str) -> str:
    normalized = str(url).strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebDownloadError(
            "INVALID_URL",
            "only absolute HTTP or HTTPS URLs are allowed",
            stage="VALIDATION",
        )
    if parsed.username or parsed.password:
        raise WebDownloadError(
            "CREDENTIALS_IN_URL",
            "URLs containing credentials are not allowed",
            stage="VALIDATION",
        )
    return normalized


def _owner_directory(root: Path, worker_id: str) -> Path:
    owner_key = hashlib.sha256(worker_id.encode("utf-8")).hexdigest()[:20]
    directory = root / f"worker-{owner_key}" / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _validate_request_destination(request):
    # Also runs on redirects; an initially public URL cannot redirect to loopback.
    from tools.local_native import _public_url
    try:
        _public_url(str(request.url))
    except (ValueError, OSError) as error:
        raise WebDownloadError("UNSAFE_DESTINATION", str(error), stage="VALIDATION") from error


def download_to_record(
    *,
    url: str,
    worker_id: str,
    tool_call_id: str,
    max_bytes: int,
    filename: str | None = None,
    root: Path = WEB_ARTIFACT_ROOT,
    transport: httpx.BaseTransport | None = None,
) -> DownloadedArtifactRecord:
    """Stream one response to disk and return runtime-authored metadata."""

    normalized_url = _validate_url(url)
    if not 1 <= max_bytes <= TASK_FILE_MAX_BYTES:
        raise ValueError("max_bytes must be between 1 and 20 MiB")
    destination = _owner_directory(root, worker_id)
    temporary = destination / f".{uuid4().hex}.part"
    digest = hashlib.sha256()
    size = 0
    media_type: str | None = None

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, read=60.0),
            headers={"User-Agent": "PersonalOps-WebWorker/1.0"},
            transport=transport,
            trust_env=False,
            event_hooks={"request": [_validate_request_destination]},
        ) as client:
            with client.stream("GET", normalized_url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        announced_size = int(content_length)
                    except ValueError:
                        announced_size = 0
                    if announced_size > max_bytes:
                        raise WebDownloadError(
                            "FILE_TOO_LARGE",
                            f"download exceeds the {max_bytes}-byte file limit",
                            stage="RESPONSE_HEADERS",
                            limit_bytes=max_bytes,
                            announced_bytes=announced_size,
                        )
                media_type = response.headers.get("content-type")
                if media_type:
                    media_type = media_type.split(";", 1)[0].strip() or None
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > max_bytes:
                            raise WebDownloadError(
                                "FILE_TOO_LARGE",
                                f"download exceeds the {max_bytes}-byte file limit",
                                stage="STREAM",
                                limit_bytes=max_bytes,
                                observed_bytes=size,
                            )
                        digest.update(chunk)
                        output.write(chunk)
                if (Path(_safe_filename(normalized_url, filename)).suffix.lower() == ".pdf"
                        or media_type == "application/pdf"):
                    with temporary.open("rb") as source:
                        signature = source.read(1024)
                    if b"%PDF-" not in signature or media_type == "text/html":
                        raise WebDownloadError("UNEXPECTED_CONTENT", "Expected PDF but received non-PDF content (possibly a login/error page)", stage="STREAM")
    except WebDownloadError:
        temporary.unlink(missing_ok=True)
        raise
    except httpx.HTTPError as error:
        temporary.unlink(missing_ok=True)
        raise WebDownloadError(
            "HTTP_ERROR",
            str(error),
            stage="STREAM",
            observed_bytes=size,
            retryable=(not isinstance(error, httpx.HTTPStatusError)
                       or error.response.status_code in {408, 429}
                       or error.response.status_code >= 500),
        ) from error
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise WebDownloadError(
            "IO_ERROR",
            str(error),
            stage="WRITE",
            observed_bytes=size,
        ) from error

    sha256 = digest.hexdigest()
    safe_name = _safe_filename(normalized_url, filename)
    if "." not in safe_name and media_type:
        extension = mimetypes.guess_extension(media_type) or ""
        safe_name = f"{safe_name}{extension}"
    final_path = destination / f"{sha256[:16]}-{safe_name}"
    os.replace(temporary, final_path)
    return DownloadedArtifactRecord(
        candidate_id=f"download-{sha256[:20]}",
        source_url=normalized_url,
        storage_path=str(final_path.resolve()),
        filename=safe_name,
        size_bytes=size,
        sha256=sha256,
        media_type=media_type,
        tool_call_id=tool_call_id,
        downloaded_at=datetime.now(timezone.utc),
    )


def create_download_web_artifact_tool(
    *,
    max_file_mib: int,
    root: Path = WEB_ARTIFACT_ROOT,
):
    """Create one Web download tool with a startup-validated byte ceiling."""

    max_bytes = int(max_file_mib) * 1024 * 1024
    if not 1 <= max_file_mib <= TASK_FILE_MAX_MIB:
        raise ValueError("max_file_mib must be between 1 and 20")

    @tool(
        DOWNLOAD_WEB_ARTIFACT_NAME,
        description=load_prompt("workers/download_web_artifact_tool"),
    )
    async def download_web_artifact(
        url: str,
        runtime: ToolRuntime,
        filename: str | None = None,
    ) -> Command:
        """Download a bounded public file for later review."""

        state = runtime.state
        worker_id = str(state.get("worker_id") or "").strip()
        tool_call_id = str(runtime.tool_call_id or "").strip()
        if not worker_id or not tool_call_id:
            result = WebDownloadToolResult(
                status="FAILED",
                error=WebDownloadFailure(
                    code="MISSING_RUNTIME_IDENTITY",
                    stage="RUNTIME",
                    message="download requires worker_id and Tool Call identity",
                ),
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=result.model_dump_json(),
                            tool_call_id=tool_call_id or DOWNLOAD_WEB_ARTIFACT_NAME,
                        )
                    ]
                }
            )
        try:
            record = await asyncio.to_thread(
                download_to_record,
                url=url,
                worker_id=worker_id,
                tool_call_id=tool_call_id,
                max_bytes=max_bytes,
                filename=filename,
                root=root,
            )
            if state.get("event_id") or state.get("planning_run_id"):
                from task_files import register_file
                record = await asyncio.to_thread(
                    register_file, record.storage_path, source_root=root, state=state,
                    tool_call_id=tool_call_id, origin="WEB", source_ref=record.source_url,
                )
        except WebDownloadError as error:
            result = WebDownloadToolResult(
                status=(
                    "REJECTED"
                    if error.failure.code in {
                        "INVALID_URL",
                        "CREDENTIALS_IN_URL",
                        "FILE_TOO_LARGE",
                        "UNSAFE_DESTINATION",
                        "UNEXPECTED_CONTENT",
                    }
                    else "FAILED"
                ),
                source_url=str(url).strip() or None,
                error=error.failure,
            )
            result_value = result.model_dump(mode="json")
            runtime.stream_writer(
                {"type": "web_download_result", "result": result_value}
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=json.dumps(result_value, ensure_ascii=False),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        record_value = record.model_dump(mode="json")
        result = WebDownloadToolResult(
            status="DOWNLOADED",
            artifact_candidate=WorkerArtifactCandidate(
                candidate_id=record.candidate_id,
                kind="DOWNLOADED_FILE",
                description=f"Downloaded {record.filename}",
                evidence_tool_call_ids=[record.tool_call_id],
            ),
            filename=record.filename,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            source_url=record.source_url,
        )
        model_result = result.model_dump(mode="json")
        model_result.update(
            reading_path=f"/downloads/{record.candidate_id}",
            reading_hint=(
                "Use read_file(reading_path) for local parsing and automatic OCR; "
                "or ocr_image (image text). Pass reading_path, not a host path. "
                "Reading limit is 20 MiB; actual content type is validated by the reader."
            ),
        )
        runtime.stream_writer(
            {"type": "web_download_result", "result": model_result}
        )
        return Command(
            update={
                "worker_downloaded_artifacts": [record_value],
                "messages": [
                    ToolMessage(
                        content=json.dumps(model_result, ensure_ascii=False),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return download_web_artifact


__all__ = [
    "DOWNLOAD_WEB_ARTIFACT_NAME",
    "WEB_ARTIFACT_ROOT",
    "WebDownloadError",
    "create_download_web_artifact_tool",
    "download_to_record",
]
