"""LangChain tools for durable local reminders."""

from __future__ import annotations

from typing import Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, ConfigDict, Field

from scheduling.models import DeliveryTarget, Recurrence, ScheduleStatus, parse_run_at
from scheduling.service import ScheduleService
from scheduling.store import ScheduleNotFoundError


_SERVICE: ScheduleService | None = None


def configure_schedule_service(service: ScheduleService | None) -> None:
    global _SERVICE
    _SERVICE = service


def _service() -> ScheduleService:
    if _SERVICE is None or not _SERVICE.running:
        raise ToolException("本地提醒服务尚未启动，未创建或修改任何提醒。")
    return _SERVICE


class CreateScheduleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(
        min_length=1,
        max_length=120,
        description="Windows通知的短标题，例如“开会提醒”或“服药提醒”。",
    )
    message: str = Field(
        min_length=1,
        max_length=1000,
        description="到期时原样显示的提醒正文；不得擅自加入用户没有要求的动作。",
    )
    run_at: str = Field(
        min_length=20,
        max_length=64,
        description=(
            "第一次触发时间，必须是带UTC偏移的ISO-8601绝对时间，"
            "例如2026-09-07T09:00:00+08:00。用户说“明天”等相对时间时，"
            "先调用get_current_time确定日期；禁止提交无时区时间。"
        ),
    )
    timezone_name: str = Field(
        default="Asia/Shanghai",
        min_length=1,
        max_length=64,
        description="IANA时区；重复提醒按该时区保持相同本地钟点，例如Asia/Shanghai。",
    )
    recurrence: Literal["ONCE", "DAILY", "WEEKLY"] = Field(
        default="ONCE",
        description="ONCE只提醒一次；DAILY每天同一当地时间；WEEKLY每周同一星期和时间。",
    )


class CreateFeishuReminderInput(CreateScheduleInput):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    runtime: ToolRuntime


class CreateAgentTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    title: str = Field(
        min_length=1,
        max_length=120,
        description="定时任务的简短标题，供用户查询和审计。",
    )
    task: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "到期后原样进入Agent队列的完整任务指令，例如“查询明天上海天气并总结”。"
            "不要加入用户未授权的发送、删除或购买动作。"
        ),
    )
    run_at: str = Field(
        min_length=20,
        max_length=64,
        description=(
            "触发时间，必须是带UTC偏移的ISO-8601绝对时间，例如"
            "2026-09-07T09:00:00+08:00。相对时间须先调用get_current_time。"
        ),
    )
    timezone_name: str = Field(
        default="Asia/Shanghai",
        min_length=1,
        max_length=64,
        description="IANA时区；当前定时Agent任务只执行一次。",
    )
    runtime: ToolRuntime


def _runtime_event_id(runtime: ToolRuntime) -> str:
    event_id = str(
        runtime.state.get("event_id") or runtime.state.get("planning_run_id") or ""
    ).strip()
    if not event_id:
        raise ToolException("当前调用没有可信Event上下文，未创建飞书日程。")
    return event_id


@tool(args_schema=CreateScheduleInput)
async def schedule_create(
    title: str,
    message: str,
    run_at: str,
    timezone_name: str = "Asia/Shanghai",
    recurrence: Literal["ONCE", "DAILY", "WEEKLY"] = "ONCE",
) -> dict:
    """创建一个持久化本地提醒。适用于“明天九点提醒我”“每周一提醒交周报”。必须把相对日期换成带时区的绝对run_at；普通提醒到期后直接交给Windows，不调用模型。返回schedule_id和规范化时间才算创建成功。"""
    try:
        service = _service()
        schedule = await service.store.create(
            title=title,
            message=message,
            run_at=parse_run_at(run_at),
            timezone_name=timezone_name,
            recurrence=Recurrence(recurrence),
        )
        service.notify_changed()
        return schedule.to_dict()
    except (ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


@tool(args_schema=CreateFeishuReminderInput)
async def schedule_create_feishu_reminder(
    title: str,
    message: str,
    run_at: str,
    runtime: ToolRuntime,
    timezone_name: str = "Asia/Shanghai",
    recurrence: Literal["ONCE", "DAILY", "WEEKLY"] = "ONCE",
) -> dict:
    """创建持久化飞书文本提醒。到期后宿主直接把message发到发起请求的当前飞书会话，不调用模型，也不允许模型指定chat_id。适用于“明天在飞书提醒我”和每日/每周飞书提醒。"""
    try:
        service = _service()
        conversation_id, reply_target_id = await service.resolve_context(
            _runtime_event_id(runtime)
        )
        schedule = await service.store.create(
            title=title,
            message=message,
            run_at=parse_run_at(run_at),
            timezone_name=timezone_name,
            recurrence=Recurrence(recurrence),
            delivery_target=DeliveryTarget.FEISHU,
            reply_target_id=reply_target_id,
            conversation_id=conversation_id,
        )
        service.notify_changed()
        return schedule.to_dict()
    except (ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


@tool(args_schema=CreateAgentTaskInput)
async def schedule_create_agent_task(
    title: str,
    task: str,
    run_at: str,
    runtime: ToolRuntime,
    timezone_name: str = "Asia/Shanghai",
) -> dict:
    """创建一次性定时Agent任务。到期后宿主将完整task作为新的SYSTEM/QUEUE Event持久化并唤醒Agent；它不会插入、替换或打断当时正在运行的任务。适用于“明天九点帮我查天气并总结”等需要届时执行能力而不只是提醒的请求。"""
    try:
        service = _service()
        conversation_id, reply_target_id = await service.resolve_context(
            _runtime_event_id(runtime)
        )
        schedule = await service.store.create(
            title=title,
            message=task,
            run_at=parse_run_at(run_at),
            timezone_name=timezone_name,
            recurrence=Recurrence.ONCE,
            delivery_target=DeliveryTarget.AGENT_EVENT,
            reply_target_id=reply_target_id,
            conversation_id=conversation_id,
        )
        service.notify_changed()
        return schedule.to_dict()
    except (ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


@tool
async def schedule_list(
    status: Literal["ALL", "ACTIVE", "PAUSED", "COMPLETED", "CANCELLED"] = "ACTIVE",
    limit: int = 50,
) -> list[dict]:
    """列出本地提醒及schedule_id、下一次触发时间、重复规则和状态。用户要取消或暂停某个提醒但没有给ID时，应先调用本工具查找，不能猜ID。status默认只看有效提醒，ALL查看历史。"""
    try:
        statuses = None if status == "ALL" else [ScheduleStatus(status)]
        items = await _service().store.list(statuses, limit=limit)
        return [item.to_dict() for item in items]
    except (ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


@tool
async def schedule_pause(schedule_id: str) -> dict:
    """暂停一个ACTIVE重复或尚未到期的本地提醒；保留定义和历史，暂停期间不会触发。必须使用schedule_list返回的真实schedule_id。"""
    try:
        result = await _service().store.set_status(schedule_id, ScheduleStatus.PAUSED)
        return result.to_dict()
    except (ScheduleNotFoundError, ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


@tool
async def schedule_resume(schedule_id: str) -> dict:
    """恢复一个PAUSED本地提醒。若原触发时间已经过去，服务会在恢复后的扫描中补一次提醒；重复任务随后跳到下一有效周期。"""
    try:
        service = _service()
        result = await service.store.set_status(schedule_id, ScheduleStatus.ACTIVE)
        service.notify_changed()
        return result.to_dict()
    except (ScheduleNotFoundError, ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


@tool
async def schedule_delete(schedule_id: str) -> dict:
    """取消一个尚未结束的本地提醒。这是保留审计记录的软删除，取消后不能恢复；只在用户明确要求删除或取消该提醒时调用，不能根据沉默或推测自行取消。"""
    try:
        result = await _service().store.set_status(schedule_id, ScheduleStatus.CANCELLED)
        return result.to_dict()
    except (ScheduleNotFoundError, ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


@tool
async def schedule_runs(schedule_id: str, limit: int = 20) -> list[dict]:
    """查看某个日程最近的执行记录。SUBMITTED表示已提交给对应目标：Windows已接受、飞书发送调用已成功，或Agent Event已持久化入队；不等于用户已看到或Agent任务已完成。FAILED包含提交失败原因。"""
    try:
        items = await _service().store.list_runs(schedule_id, limit=limit)
        return [item.to_dict() for item in items]
    except (ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


@tool
async def windows_notify(title: str, message: str) -> dict:
    """立即提交一条Windows右下角原生通知，不创建日程。用于用户明确要求“现在弹个通知”或验证通知功能；普通未来提醒应使用schedule_create。成功仅表示Windows接受请求，专注助手或系统设置仍可能隐藏它。"""
    try:
        return await _service().submit_notification(title=title, message=message)
    except (ValueError, RuntimeError) as error:
        raise ToolException(str(error)) from error


SCHEDULE_TOOLS = [
    schedule_create,
    schedule_create_feishu_reminder,
    schedule_create_agent_task,
    schedule_list,
    schedule_pause,
    schedule_resume,
    schedule_delete,
    schedule_runs,
    windows_notify,
]

for current_tool in SCHEDULE_TOOLS:
    current_tool.handle_tool_error = True
    current_tool.metadata = {
        "capability": "scheduled_automation",
        "execution": "local_host",
        "network": "none",
        "delivery": "windows_native_notification",
        "external_side_effect": False,
    }

schedule_create_feishu_reminder.metadata = {
    **schedule_create_feishu_reminder.metadata,
    "network": "feishu",
    "delivery": "authenticated_current_chat",
    "external_side_effect": True,
    "recipient_binding": "trusted_event_context",
}

schedule_create_agent_task.metadata = {
    **schedule_create_agent_task.metadata,
    "network": "deferred_by_agent_tools",
    "delivery": "durable_agent_event_queue",
    "external_side_effect": True,
    "queue_action": "QUEUE",
    "recurrence": "ONCE",
    "recipient_binding": "trusted_event_context",
}

schedule_delete.metadata = {
    **schedule_delete.metadata,
    "risk": "local_schedule_cancel",
    "requires_explicit_user_intent": True,
    "recoverability": "definition retained as CANCELLED for audit",
}
