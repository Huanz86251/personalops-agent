"""A native LangChain tool; privileged delivery stays in the host harness."""

import asyncio
import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, ConfigDict, Field
from deepagents.backends.utils import file_data_to_string

from artifact_models import DownloadedArtifactRecord

from feishu_exports import (
    ExportDenied,
    ExportUncertain,
    current_export_identity,
    request_export,
)
from host_file_access import (
    HOST_FILE_ACCESS,
    HostFileApprovalRequest,
    HostFileApprovalUnavailable,
    HostFileUserDenied,
)
from run_workspace import initialize_run_workspace, materialize_worker_artifact
from task_files import read_download


class SendLocalFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    path: str = Field(
        min_length=1,
        max_length=2048,
        description=(
            "要交付的精确文件引用：可使用本机绝对路径、当前Worker的"
            "/downloads/<candidate_id>、已审核共享的/handoff/...，或当前工作区文件路径。"
            "文件直接发送，目录打包为ZIP。"
        ),
    )
    reason: str = Field(
        min_length=1,
        max_length=1200,
        description="说明为什么当前用户原始请求和当前Step需要把这个精确文件发送给用户。",
    )
    runtime: ToolRuntime


def _state_file_bytes(value) -> bytes:
    content = file_data_to_string(value)
    if isinstance(value, dict) and str(value.get("encoding", "utf-8")) == "base64":
        return base64.standard_b64decode(content)
    return content.encode("utf-8")


def _safe_virtual_relative(path: str, prefix: str) -> Path:
    relative = path.removeprefix(prefix).replace("\\", "/").lstrip("/")
    parts = [part for part in relative.split("/") if part]
    if not parts or ".." in parts:
        raise ExportDenied("任务文件引用无效或试图越出受控目录。")
    return Path(*parts)


@dataclass(frozen=True)
class _ResolvedDeliverySource:
    physical_path: Path
    trusted_path_mapping: str


def _compact_mapping(value: str, limit: int = 6000) -> str:
    normalized = " ".join(str(value).strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _download_source_label(record: DownloadedArtifactRecord) -> str:
    if record.origin == "HOST" and record.source_url.startswith("host:"):
        source = record.source_url.removeprefix("host:").strip()
        if Path(source).is_absolute():
            return f"原始本机路径 `{source}`"
        return f"旧版主机来源标识 `{record.source_url}`"
    if record.origin in {"WEB", "BROWSER"}:
        return f"网络来源 `{record.source_url}`"
    if record.origin == "EMAIL":
        return f"邮件附件来源 `{record.source_url}`"
    return f"已登记来源 `{record.source_url}`"


def _resolve_delivery_source_with_mapping(
    path: str, state: dict
) -> _ResolvedDeliverySource:
    """Resolve one source and construct Harness-authored provenance for Judge."""

    normalized = str(path).strip().replace("\\", "/")
    if normalized.startswith("/downloads/"):
        payload = read_download(normalized, state)
        candidate_id = normalized.removeprefix("/downloads/")
        raw_record = next(
            (
                item
                for item in reversed(state.get("worker_downloaded_artifacts") or [])
                if str(item.get("candidate_id") or "") == candidate_id
            ),
            None,
        )
        if raw_record is None:
            raise ExportDenied("下载候选不属于当前Worker。")
        record = DownloadedArtifactRecord.model_validate(raw_record)
        physical = Path(record.storage_path).resolve(strict=True)
        # read_download already revalidated owner, size and digest. Keep the
        # payload read in the path to prevent bypassing that validation.
        if len(payload) != physical.stat().st_size:
            raise ExportDenied("下载候选的物理文件与登记大小不一致。")
        mapping = (
            f"{_download_source_label(record)} -> Worker可见引用 `{normalized}` -> "
            f"已校验隔离副本 `{physical}`；candidate_id、大小与SHA-256均已由Harness复核。"
        )
        return _ResolvedDeliverySource(physical, _compact_mapping(mapping))

    physical = _resolve_delivery_source(path, state)
    if normalized.startswith("/handoff/"):
        mapping = (
            f"已审核共享引用 `{normalized}` -> 当前任务handoff实体 `{physical}`；"
            "路径边界与普通文件类型已由Harness复核。"
        )
    elif (state.get("files") or {}).get(path) is not None or (
        state.get("files") or {}
    ).get(normalized) is not None:
        mapping = (
            f"Worker工作区引用 `{normalized}` -> 本次发送冻结副本 `{physical}`；"
            "内容已由Harness从当前Worker状态物化。"
        )
    else:
        mapping = (
            f"原始本机绝对路径 `{physical}` -> 本次发送源 `{physical}`；"
            "存在性与规范化路径已由Harness复核。"
        )
    return _ResolvedDeliverySource(physical, _compact_mapping(mapping))


def _resolve_delivery_source(path: str, state: dict) -> Path:
    """Resolve model-visible file references to one validated host file."""

    normalized = str(path).strip().replace("\\", "/")
    if normalized.startswith("/downloads/"):
        payload = read_download(normalized, state)
        candidate_id = normalized.removeprefix("/downloads/")
        raw_record = next(
            (
                item
                for item in reversed(state.get("worker_downloaded_artifacts") or [])
                if str(item.get("candidate_id") or "") == candidate_id
            ),
            None,
        )
        if raw_record is None:
            raise ExportDenied("下载候选不属于当前Worker。")
        physical = Path(str(raw_record.get("storage_path") or "")).resolve(strict=True)
        # read_download already revalidated owner, size and digest.  Keep the
        # payload read in the path to prevent bypassing that validation.
        if len(payload) != physical.stat().st_size:
            raise ExportDenied("下载候选的物理文件与登记大小不一致。")
        return physical

    run_id = str(state.get("event_id") or state.get("planning_run_id") or "").strip()
    worker_id = str(state.get("worker_id") or "").strip()
    storage_root = state.get("run_storage_root")
    if normalized.startswith("/handoff/"):
        if not run_id:
            raise ExportDenied("当前任务缺少run身份，无法解析handoff文件。")
        kwargs = {"storage_root": Path(storage_root)} if storage_root else {}
        layout = initialize_run_workspace(run_id, **kwargs)
        physical = (layout.handoff_root / _safe_virtual_relative(normalized, "/handoff/")).resolve(
            strict=True
        )
        try:
            physical.relative_to(layout.handoff_root.resolve())
        except ValueError as error:
            raise ExportDenied("handoff文件引用越出当前任务目录。") from error
        if not physical.is_file():
            raise ExportDenied("handoff引用不是可发送的普通文件。")
        return physical

    files = state.get("files") or {}
    file_value = files.get(path)
    if file_value is None:
        file_value = files.get(normalized)
    if file_value is not None:
        if not run_id or not worker_id:
            raise ExportDenied("当前任务缺少run或Worker身份，无法冻结工作区文件。")
        payload = _state_file_bytes(file_value)
        if not payload:
            raise ExportDenied("不能发送空的工作区文件。")
        kwargs = {"storage_root": Path(storage_root)} if storage_root else {}
        layout = initialize_run_workspace(run_id, **kwargs)
        filename = Path(normalized).name or "artifact.bin"
        candidate_id = "delivery-" + hashlib.sha256(
            (normalized + "\0").encode("utf-8") + payload
        ).hexdigest()[:24]
        return materialize_worker_artifact(
            layout=layout,
            worker_id=worker_id,
            candidate_id=candidate_id,
            filename=filename,
            payload=payload,
        ).resolve(strict=True)

    physical = Path(path)
    if not physical.is_absolute():
        raise ExportDenied(
            "无法解析文件引用；请使用本机绝对路径、/downloads、/handoff或当前工作区路径。"
        )
    return physical.resolve(strict=True)


@tool(args_schema=SendLocalFileInput)
async def send_local_file_to_feishu(path: str, reason: str, runtime: ToolRuntime) -> dict:
    """把精确文件发回当前飞书会话；所有外发都由独立Judge审批。"""
    try:
        state = runtime.state
        event_id = str(
            state.get("event_id") or state.get("planning_run_id") or ""
        )
        identity = current_export_identity()
        resolved_source = await asyncio.to_thread(
            _resolve_delivery_source_with_mapping, path, state
        )
        await HOST_FILE_ACCESS.authorize(
            HostFileApprovalRequest(
                event_id=event_id,
                chat_id=identity.chat_id if identity is not None else "",
                worker_id=str(state.get("worker_id") or "").strip(),
                worker_kind=str(state.get("host_worker_kind") or "GENERAL").strip(),
                operation="SEND_TO_USER",
                path=path,
                worker_reason=reason,
                original_user_request=str(state.get("host_original_user_request") or "").strip(),
                current_step=str(state.get("host_current_step") or "").strip(),
                trusted_path_mapping=resolved_source.trusted_path_mapping,
            )
        )
        return await request_export(str(resolved_source.physical_path), event_id)
    except HostFileUserDenied as error:
        return {
            "status": "USER_DENIED",
            "code": "USER_DENIED",
            "message": str(error),
            "path": path,
            "retryable": False,
            "required_next_action": (
                "不得重试或绕过；不得解释成文件不存在、工具缺失或普通失败。"
            ),
        }
    except HostFileApprovalUnavailable as error:
        return {
            "status": "APPROVAL_UNAVAILABLE",
            "code": error.code,
            "message": str(error),
            "path": path,
            "retryable": False,
        }
    except (ExportDenied, ExportUncertain, OSError) as error:
        raise ToolException(str(error)) from error


send_local_file_to_feishu.handle_tool_error = True
send_local_file_to_feishu.metadata = {
    "capability": "feishu_file_export",
    "risk": "external_file_send",
    "request_role": "worker",
    "executor": "host_after_policy_checks",
    "recipient": "authenticated_current_chat",
    "approval": "judge_then_human_if_uncertain",
}
