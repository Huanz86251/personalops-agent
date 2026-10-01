import asyncio
import json
import unittest

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from hard_planning import _invoke_structured
from planning_models import PlanStep
from step_user_question import (
    STEP_USER_QUESTIONS,
    StepUserQuestionRequest,
    StepUserQuestionService,
    active_ask_user_interactions,
    ask_user_runtime_scope,
    create_ask_user_tool,
    invoke_with_ask_user,
)


class StepUserQuestionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        STEP_USER_QUESTIONS.configure_notifier(None)

    async def test_service_delivers_freeform_answer_to_same_waiter(self):
        service = StepUserQuestionService()
        notified = []

        async def notify(chat_id, request):
            notified.append((chat_id, request.question))
            asyncio.get_running_loop().call_soon(
                service.consume_user_reply,
                chat_id,
                "请使用 E:\\Downloads 里的最终版简历",
            )

        service.configure_notifier(notify)
        answer = await service.ask(
            StepUserQuestionRequest(
                event_id="event-1",
                chat_id="chat-1",
                role="GENERAL_WORKER",
                question="应该使用哪一份简历？",
            )
        )
        self.assertEqual(notified, [("chat-1", "应该使用哪一份简历？")])
        self.assertEqual(answer, "请使用 E:\\Downloads 里的最终版简历")
        self.assertEqual(service.pending_requests, ())

    async def test_unrelated_chat_is_not_consumed(self):
        service = StepUserQuestionService()
        self.assertFalse(service.consume_user_reply("other-chat", "答案").handled)

    async def test_pending_state_is_keyed_by_real_reply_target(self):
        service = StepUserQuestionService()
        notified = asyncio.Event()

        async def notify(chat_id, request):
            notified.set()

        service.configure_notifier(notify)
        task = asyncio.create_task(
            service.ask(
                StepUserQuestionRequest(
                    event_id="event-1",
                    chat_id="oc_real_chat",
                    role="FINAL_REVIEWER",
                    question="请提供文件名。",
                )
            )
        )
        await notified.wait()
        self.assertTrue(service.has_pending("oc_real_chat"))
        self.assertFalse(service.has_pending("owner"))
        service.consume_user_reply("oc_real_chat", "resume.pdf")
        self.assertEqual(await task, "resume.pdf")
        self.assertFalse(service.has_pending("oc_real_chat"))

    async def test_tool_waits_and_records_answer_in_current_runtime(self):
        async def notify(chat_id, request):
            self.assertEqual(request.role, "WEB_WORKER")
            asyncio.get_running_loop().call_soon(
                STEP_USER_QUESTIONS.consume_user_reply,
                chat_id,
                "选择第二项",
            )

        STEP_USER_QUESTIONS.configure_notifier(notify)
        with ask_user_runtime_scope(event_id="event-2", chat_id="chat-2"):
            result = json.loads(
                await create_ask_user_tool("WEB_WORKER").ainvoke(
                    {"question": "两个候选中选择哪一个？"}
                )
            )
            interactions = active_ask_user_interactions()

        self.assertEqual(result["status"], "ANSWERED")
        self.assertEqual(result["answer"], "选择第二项")
        self.assertEqual(interactions[0]["role"], "WEB_WORKER")

    async def test_plain_role_continues_after_tool_answer(self):
        class Model:
            def __init__(self):
                self.calls = 0
                self.last_messages = []

            def bind_tools(self, tools, **kwargs):
                self.tool_names = [tool.name for tool in tools]
                return self

            async def ainvoke(self, messages, config=None):
                self.calls += 1
                self.last_messages = list(messages)
                if self.calls == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "id": "ask-1",
                            "name": "ask_user",
                            "args": {"question": "使用正式版吗？"},
                        }],
                    )
                return AIMessage(content="已按正式版继续处理。")

        async def notify(chat_id, request):
            asyncio.get_running_loop().call_soon(
                STEP_USER_QUESTIONS.consume_user_reply,
                chat_id,
                "是，使用正式版",
            )

        STEP_USER_QUESTIONS.configure_notifier(notify)
        model = Model()
        with ask_user_runtime_scope(event_id="event-3", chat_id="chat-3"):
            response, rounds = await invoke_with_ask_user(
                model,
                [{"role": "user", "content": "处理简历"}],
                role="SCHEDULER_DIRECT_ANSWER",
            )

        self.assertEqual(rounds, 2)
        self.assertEqual(response.content, "已按正式版继续处理。")
        self.assertEqual(model.tool_names, ["ask_user"])
        self.assertTrue(any(isinstance(item, ToolMessage) for item in model.last_messages))

    async def test_structured_scheduler_role_can_ask_then_submit_schema(self):
        class Decision(BaseModel):
            answer: str

        class Model:
            def __init__(self):
                self.calls = 0
                self.bound_names = []

            def bind_tools(self, tools, **kwargs):
                self.bound_names = [
                    getattr(tool, "name", getattr(tool, "__name__", ""))
                    for tool in tools
                ]
                return self

            def with_structured_output(self, *args, **kwargs):
                raise AssertionError("Shared Human Loop should use the tool path")

            async def ainvoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return AIMessage(content="", tool_calls=[{
                        "id": "ask-hard-1",
                        "name": "ask_user",
                        "args": {"question": "是否使用正式版本？"},
                    }])
                self.asserted_tool_answer = any(
                    isinstance(item, ToolMessage) and "使用正式版本" in item.content
                    for item in messages
                )
                return AIMessage(content="", tool_calls=[{
                    "id": "decision-1",
                    "name": "Decision",
                    "args": {"answer": "继续正式版本"},
                }])

        async def notify(chat_id, request):
            asyncio.get_running_loop().call_soon(
                STEP_USER_QUESTIONS.consume_user_reply,
                chat_id,
                "使用正式版本",
            )

        STEP_USER_QUESTIONS.configure_notifier(notify)
        model = Model()
        with ask_user_runtime_scope(event_id="event-4", chat_id="chat-4"):
            result = await _invoke_structured(
                model,
                prompt="请选择。",
                output_schema=Decision,
                trace_name="hard_final_reviewer",
                fallback_factory=lambda error: Decision(answer="fallback"),
            )

        self.assertEqual(model.calls, 2)
        self.assertEqual(model.bound_names, ["ask_user", "Decision"])
        self.assertTrue(model.asserted_tool_answer)
        self.assertEqual(result.output.answer, "继续正式版本")
        self.assertEqual(result.model_rounds_used, 2)

    def test_scheduler_step_schema_no_longer_contains_question_field(self):
        self.assertNotIn("ask_user_question", PlanStep.model_json_schema()["properties"])


if __name__ == "__main__":
    unittest.main()
