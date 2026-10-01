"""Shared Human Loop used by every execution/review role.

The model-facing ``ask_user`` tool deliberately waits inside the current
role invocation.  An inbound Feishu reply is consumed by the service before
it can become a new Agent event, then returned as the tool result so the same
role continues with its existing message history.

"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any

from langchain_core.tools import BaseTool, tool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from observability import set_span_output, trace_span
from prompt_loader import load_prompt


ASK_USER_NAME = "ask_user"


class StepUserQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    step_id: int = Field(default=0, ge=0)
    objective: str = Field(default="当前角色需要用户补充信息。", min_length=1, max_length=6000)
    role: str = Field(default="AGENT", min_length=1, max_length=120)
    question: str = Field(min_length=1, max_length=1200)


class AskUserInput(BaseModel):
    """The only model-authored argument for the shared Human Loop."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "要直接展示给用户的一个具体问题。只问继续当前任务所必需、且工具或现有"
            "上下文无法确定的信息；不要把多个不相关问题合并。"
        ),
    )


@dataclass(frozen=True)
class HumanReplyResult:
    handled: bool
    reply: str = ""


QuestionNotifier = Callable[[str, StepUserQuestionRequest], Awaitable[None]]


@dataclass
class AskUserRuntimeContext:
    event_id: str
    chat_id: str
    interactions: list[dict[str, Any]]


ACTIVE_ASK_USER_CONTEXT: ContextVar[AskUserRuntimeContext | None] = ContextVar(
    "active_ask_user_context",
    default=None,
)


class StepUserQuestionService:
    def __init__(self) -> None:
        self.notifier: QuestionNotifier | None = None
        self._pending: dict[str, asyncio.Future[str]] = {}
        self._requests: dict[str, StepUserQuestionRequest] = {}

    def configure_notifier(self, notifier: QuestionNotifier | None) -> None:
        self.notifier = notifier

    async def ask(self, request: StepUserQuestionRequest) -> str:
        if self.notifier is None:
            raise RuntimeError("当前渠道无法发送用户问题。")
        if request.chat_id in self._pending:
            raise RuntimeError("当前会话已有一个问题等待用户回答。")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending[request.chat_id] = future
        self._requests[request.chat_id] = request
        try:
            await self.notifier(request.chat_id, request)
            return await future
        finally:
            self._pending.pop(request.chat_id, None)
            self._requests.pop(request.chat_id, None)

    def consume_user_reply(self, chat_id: str, text: str) -> HumanReplyResult:
        future = self._pending.get(chat_id)
        if future is None:
            return HumanReplyResult(False)
        answer = str(text or "").strip()
        if not answer:
            return HumanReplyResult(True, "当前正在等待回答，请直接回复问题所需的信息。")
        if not future.done():
            future.set_result(answer)
        return HumanReplyResult(True, "已收到回答，正在继续当前步骤。")

    def has_pending(self, chat_id: str | None = None) -> bool:
        """Return whether execution is intentionally waiting for a human answer.

        ``chat_id`` is the concrete channel reply target, not the stable
        Conversation owner key. Keeping those identities separate prevents a
        question from being sent to the synthetic ``owner`` conversation id.
        """

        if chat_id is None:
            return bool(self._pending)
        return str(chat_id).strip() in self._pending

    @property
    def pending_requests(self) -> tuple[StepUserQuestionRequest, ...]:
        return tuple(self._requests.values())


STEP_USER_QUESTIONS = StepUserQuestionService()


@contextmanager
def ask_user_runtime_scope(*, event_id: str, chat_id: str):
    """Bind the durable channel target to all child model/tool calls."""

    context = AskUserRuntimeContext(
        event_id=str(event_id or "").strip(),
        chat_id=str(chat_id or "").strip(),
        interactions=[],
    )
    if not context.event_id or not context.chat_id:
        raise ValueError("ask_user runtime requires event_id and chat_id.")
    token = ACTIVE_ASK_USER_CONTEXT.set(context)
    try:
        yield context
    finally:
        ACTIVE_ASK_USER_CONTEXT.reset(token)


def ask_user_available() -> bool:
    context = ACTIVE_ASK_USER_CONTEXT.get()
    return bool(context is not None and context.chat_id and STEP_USER_QUESTIONS.notifier)


def active_ask_user_interactions() -> list[dict[str, Any]]:
    context = ACTIVE_ASK_USER_CONTEXT.get()
    return list(context.interactions) if context is not None else []


def merge_ask_user_interactions(
    existing: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> list[dict[str, Any]]:
    """Append current-run answers without duplicating checkpointed records."""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*(existing or []), *active_ask_user_interactions()]:
        if not isinstance(item, dict):
            continue
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
    return merged


def create_ask_user_tool(role: str) -> BaseTool:
    """Create the same named control tool with role-specific audit metadata."""

    normalized_role = str(role or "AGENT").strip().upper() or "AGENT"

    @tool(
        ASK_USER_NAME,
        args_schema=AskUserInput,
        description=load_prompt("tools/ask_user"),
    )
    async def ask_user(question: str) -> str:
        context = ACTIVE_ASK_USER_CONTEXT.get()
        if context is None or not context.chat_id:
            raise RuntimeError("当前调用没有可恢复的用户会话，不能询问用户。")

        asked_at = datetime.now(timezone.utc).isoformat()
        request = StepUserQuestionRequest(
            event_id=context.event_id,
            chat_id=context.chat_id,
            role=normalized_role,
            question=question,
            objective=f"{normalized_role} 正在执行当前任务。",
        )
        with trace_span(
            "Human Loop / Ask User",
            kind="tool",
            input_value={"role": normalized_role, "question": question},
        ) as span:
            answer = await STEP_USER_QUESTIONS.ask(request)
            record = {
                "role": normalized_role,
                "question": question.strip(),
                "answer": answer,
                "asked_at": asked_at,
                "answered_at": datetime.now(timezone.utc).isoformat(),
            }
            context.interactions.append(record)
            result = {
                "status": "ANSWERED",
                "question": question.strip(),
                "answer": answer,
                "instruction": (
                    "用户回答已经返回当前角色。直接依据该回答继续原任务；"
                    "答案清楚时不要再次确认，也不要把它当成新的用户任务。"
                ),
            }
            set_span_output(span, result)
            return json.dumps(result, ensure_ascii=False)

    ask_user.metadata = {
        "capability": "human_loop",
        "control_plane": True,
        "role": normalized_role,
        "counts_as_business_tool": False,
    }
    return ask_user


async def invoke_with_ask_user(
    model: Any,
    messages: list[Any],
    *,
    role: str,
    config: Any = None,
    max_questions: int = 3,
) -> tuple[Any, int]:
    """Invoke a plain chat role while allowing an in-place Human Loop.

    Unlike the removed Step-level planning field, the answer is appended after
    the tool result and the exact same role/model invocation continues.
    """

    conversation = list(messages)
    rounds = 0
    if not ask_user_available() or not hasattr(model, "bind_tools"):
        response = await model.ainvoke(conversation, config=config)
        return response, 1

    ask_tool = create_ask_user_tool(role)
    try:
        runnable = model.bind_tools(
            [ask_tool],
            parallel_tool_calls=False,
        )
    except TypeError:
        runnable = model.bind_tools([ask_tool])

    questions_used = 0
    while True:
        rounds += 1
        response = await runnable.ainvoke(conversation, config=config)
        calls = [
            call
            for call in list(getattr(response, "tool_calls", []) or [])
            if str(call.get("name") or "") == ASK_USER_NAME
        ]
        if not calls:
            return response, rounds
        if questions_used >= max_questions:
            raise RuntimeError(
                f"{role} 连续询问用户已达到{max_questions}次，不能继续询问。"
            )

        call = calls[0]
        additional_kwargs = dict(
            getattr(response, "additional_kwargs", {}) or {}
        )
        raw_calls = additional_kwargs.get("tool_calls")
        if isinstance(raw_calls, list):
            additional_kwargs["tool_calls"] = [
                item
                for item in raw_calls
                if isinstance(item, dict)
                and str(item.get("id") or "") == str(call.get("id") or "")
            ]
        response = response.model_copy(
            update={
                "tool_calls": [call],
                "additional_kwargs": additional_kwargs,
            }
        )
        conversation.append(response)
        answer = await ask_tool.ainvoke(call.get("args") or {})
        conversation.append(
            ToolMessage(
                content=str(answer),
                tool_call_id=str(call.get("id") or ASK_USER_NAME),
                name=ASK_USER_NAME,
            )
        )
        questions_used += 1


def configure_step_user_question_notifier(
    notifier: QuestionNotifier | None,
) -> None:
    STEP_USER_QUESTIONS.configure_notifier(notifier)


__all__ = [
    "ACTIVE_ASK_USER_CONTEXT",
    "ASK_USER_NAME",
    "AskUserInput",
    "AskUserRuntimeContext",
    "HumanReplyResult",
    "STEP_USER_QUESTIONS",
    "StepUserQuestionRequest",
    "StepUserQuestionService",
    "active_ask_user_interactions",
    "ask_user_available",
    "ask_user_runtime_scope",
    "configure_step_user_question_notifier",
    "create_ask_user_tool",
    "invoke_with_ask_user",
    "merge_ask_user_interactions",
]
