"""A native LangChain tool; privileged delivery stays in the host harness."""

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, ConfigDict, Field

from feishu_exports import ExportDenied, request_export


class SendLocalFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    path: str = Field(
        min_length=1,
        max_length=2048,
        description="允许导出目录内的本地绝对路径；文件直接发送，目录打包为 ZIP。",
    )
    runtime: ToolRuntime


@tool(args_schema=SendLocalFileInput)
async def send_local_file_to_feishu(path: str, runtime: ToolRuntime) -> dict:
    """申请把本地文件或目录发回当前飞书会话。仅授权用户的 General 可用；先发清单等待用户确认，不能自行批准。单个发送文件最多30 MB；目录最多100文件、原始合计100 MB。PENDING不是已发送，SENT及message_id才是发送回执。"""
    try:
        event_id = str(
            runtime.state.get("event_id") or runtime.state.get("planning_run_id") or ""
        )
        return await request_export(path, event_id)
    except (ExportDenied, OSError) as error:
        raise ToolException(str(error)) from error


send_local_file_to_feishu.handle_tool_error = True
send_local_file_to_feishu.metadata = {
    "capability": "feishu_file_export",
    "risk": "external_file_send",
    "request_role": "general",
    "executor": "host_after_owner_confirmation",
    "recipient": "authenticated_current_chat",
    "approval": "required",
}
