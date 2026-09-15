"""Task file ingress and local read_file dispatch, independent of model policy."""

import asyncio
import hashlib
import json
from pathlib import Path
import re
import shlex
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langgraph.types import Command

from task_files import register_file

DOCUMENTS = {".pdf", ".docx", ".pptx", ".xlsx"}


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(item.get("text", "") for item in content if isinstance(item, dict))


def _snapshot(root):
    result = {}
    for path in Path(root).rglob("*"):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            result[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns)
        if len(result) > 5000:
            raise ValueError("Browser download directory exceeds inspection limit")
    return result


class TaskFileAccessMiddleware(AgentMiddleware):
    """Only registered adapters can introduce files; read_file never emits raw PDF."""

    def __init__(self, *, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def _is_document(self, request):
        if request.tool_call["name"] != "read_file":
            return False
        path = str(request.tool_call["args"].get("file_path", ""))
        return path.startswith("/downloads/") or Path(path).suffix.lower() in DOCUMENTS

    def _read(self, request):
        args = request.tool_call["args"]
        path = str(args.get("file_path", ""))
        offset, limit = args.get("offset", 0), args.get("limit", 2000)
        if (
            not isinstance(offset, int)
            or offset < 0
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise ValueError("offset must be >= 0 and limit >= 1")
        if self.sandbox is not None:
            result = self.sandbox.execute(
                "python /opt/personalops/read_document.py "
                + shlex.quote(path)
                + f" --start-page {offset + 1} --max-pages {min(limit, 20)}"
            )
            output = result.output
            if result.exit_code and not output.strip():
                output = json.dumps(
                    {
                        "error": "Local document reader stopped (deadline/resource limit); retry a smaller page range",
                        "exit_code": result.exit_code,
                    }
                )
            return ToolMessage(
                content=output,
                status="error" if result.exit_code else "success",
                tool_call_id=request.tool_call["id"],
            )
        from tools.local_native import _read_attachment_result, _bounded

        # Persist complete extraction; only a bounded preview enters the model context.
        output = "/artifacts/read-" + uuid4().hex[:16] + ".md"
        return _bounded(_read_attachment_result)(
            path,
            output,
            request.runtime,
            start_page=offset + 1,
            max_pages=min(limit, 20),
            ocr="auto",
        )

    def _register(self, request, result, metadata, before):
        if not isinstance(result, ToolMessage) or result.status == "error":
            return result
        text = _text(result.content)
        root = Path(metadata["task_file_source_root"]).resolve()
        origin = metadata["task_file_origin"]
        sources = []
        if origin == "EMAIL":
            # The server's response envelope, not arbitrary email body text.
            marker = text.find("\n{")
            if marker < 0:
                raise ValueError(
                    "Attachment response has no structured download envelope"
                )
            body = json.loads(text[marker + 1 :])
            item = body.get("attachment", {})
            sources = [(Path(item["path"]), item.get("sha256"), item.get("size"))]
        elif origin == "BROWSER":
            # An event alone is insufficient: it must name a newly completed file
            # in this leased browser's output root. Existing snapshots are not imports.
            for value in re.findall(
                r'^- Downloaded file .+? to ["`](.+?)["`]\s*$', text, re.MULTILINE
            ):
                path = Path(value).resolve()
                path.relative_to(root)
                if path.is_file():
                    stat = path.stat()
                    if before.get(str(path)) != (stat.st_size, stat.st_mtime_ns):
                        sources.append((path, None, None))
        records = []
        for path, expected_sha, expected_size in sources:
            record = register_file(
                path,
                source_root=root,
                state=request.state,
                tool_call_id=request.tool_call["id"],
                origin=origin,
                source_ref=origin.lower()
                + ":"
                + hashlib.sha256(str(path).encode()).hexdigest(),
            )
            if expected_sha is not None and record.sha256 != expected_sha:
                raise ValueError("Attachment digest does not match download response")
            if expected_size is not None and record.size_bytes != expected_size:
                raise ValueError("Attachment size does not match download response")
            records.append(record)
        if not records:
            return result
        manifest = [
            {
                "candidate_id": r.candidate_id,
                "reading_path": "/downloads/" + r.candidate_id,
                "filename": r.filename,
                "size_bytes": r.size_bytes,
                "sha256": r.sha256,
                "evidence_tool_call_id": r.tool_call_id,
            }
            for r in records
        ]
        note = json.dumps(
            {
                "private_files": manifest,
                "published": False,
                "next_step": "read_file(reading_path) reads locally. Shared handoff requires independent review.",
            },
            ensure_ascii=False,
        )
        # Do not send host attachment paths back as usable task paths.
        content = note if origin == "EMAIL" else text + "\n" + note
        return Command(
            update={
                "worker_downloaded_artifacts": [
                    r.model_dump(mode="json") for r in records
                ],
                "messages": [
                    ToolMessage(content=content, tool_call_id=request.tool_call["id"])
                ],
            }
        )

    async def awrap_tool_call(self, request, handler):
        try:
            if self._is_document(request):
                return await asyncio.to_thread(self._read, request)
            metadata = (request.tool.metadata or {}) if request.tool else {}
            root = metadata.get("task_file_source_root")
            before = (
                await asyncio.to_thread(_snapshot, root)
                if root and metadata.get("task_file_origin") == "BROWSER"
                else {}
            )
            result = await handler(request)
            if root:
                return await asyncio.to_thread(
                    self._register, request, result, metadata, before
                )
            return result
        except (ValueError, OSError, KeyError, ToolException) as error:
            return ToolMessage(
                content=f"Task file operation failed: {error}. No shared publication occurred.",
                status="error",
                tool_call_id=request.tool_call["id"],
            )

    def wrap_tool_call(self, request, handler):
        if self._is_document(request):
            try:
                return self._read(request)
            except (ValueError, OSError, ToolException) as error:
                return ToolMessage(
                    content=f"Local reading failed: {error}",
                    status="error",
                    tool_call_id=request.tool_call["id"],
                )
        return handler(request)
