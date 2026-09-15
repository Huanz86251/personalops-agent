"""Offline tests for the minimal Deep Agents general worker."""

import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from workers.general_worker import (
    DEEP_AGENT_FILE_TOOLS,
    LEGACY_TOOL_NAMES,
    create_general_worker,
    select_general_worker_tools,
)


class StaticWorkerModel(BaseChatModel):
    """Provider-free model that records the tools bound by Deep Agents."""

    seen_tool_names: list[str] = Field(default_factory=list)
    bound_tool_history: list[list[str]] = Field(default_factory=list)

    def with_structured_output(self, schema, **kwargs):
        if schema.__name__ == 'SkillChoice':
            return RunnableLambda(lambda messages: {'parsed': {'skill_ids': [], 'reason': 'Offline business probe needs no skill'}})
        return super().with_structured_output(schema, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "static-general-worker-test"

    def bind_tools(self, tools, **kwargs):
        self.seen_tool_names = [tool.name for tool in tools]
        self.bound_tool_history.append(list(self.seen_tool_names))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="general worker ready")
                )
            ]
        )


@tool
def business_probe(value: str) -> str:
    """Return the supplied value as a business-tool test probe."""

    return value


class GeneralWorkerTests(unittest.TestCase):
    def test_default_selection_keeps_only_current_business_tools(self):
        selected = select_general_worker_tools()

        self.assertEqual(
            [tool.name for tool in selected],
            [
                "get_current_time",
                "attachment_to_text",
                "ocr_image",
                "convert_document",
                "spreadsheet_read",
                "spreadsheet_write",
                "spreadsheet_format",
                "spreadsheet_chart",
                "symbolic_math",
                "python_syntax_check",
                "python_static_check",
                "schedule_create",
                "schedule_create_feishu_reminder",
                "schedule_create_agent_task",
                "schedule_list",
                "schedule_pause",
                "schedule_resume",
                "schedule_delete",
                "schedule_runs",
                "windows_notify",
                "capture_desktop_screenshot",
            ],
        )

    def test_selection_rejects_duplicate_tool_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate tool name"):
            select_general_worker_tools(
                [business_probe, business_probe]
            )

    def test_graph_runs_without_provider_and_hides_shell(self):
        model = StaticWorkerModel()
        worker = create_general_worker(
            model,
            tools=[business_probe],
        )

        result = worker.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Confirm that the worker can run.",
                    }
                ]
            }
        )

        self.assertEqual(
            result["messages"][-1].content,
            "general worker ready",
        )
        initial_tools = model.bound_tool_history[0]
        self.assertIn("business_probe", initial_tools)
        self.assertTrue(
            set(DEEP_AGENT_FILE_TOOLS).issubset(initial_tools)
        )
        self.assertNotIn("execute", initial_tools)
        self.assertNotIn("show_all_toolsets", initial_tools)
        legacy_only_names = (
            LEGACY_TOOL_NAMES.difference(DEEP_AGENT_FILE_TOOLS)
            .difference({"show_all_toolsets"})
        )
        self.assertTrue(
            legacy_only_names.isdisjoint(initial_tools)
        )
        self.assertIn("report_general_result", model.seen_tool_names)
        self.assertNotIn("submit_for_review", model.seen_tool_names)
        self.assertNotIn("publish_worker_progress", model.seen_tool_names)
        self.assertNotIn("task", model.seen_tool_names)
        self.assertEqual(len(model.bound_tool_history), 1)


class GeneralWorkerCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_deep_agent_graph_persists_conversation_metadata(self):
        worker = create_general_worker(
            StaticWorkerModel(),
            tools=[business_probe],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "conversation-worker-1"}}

        await worker.aupdate_state(
            config,
            {
                "messages": [],
                "conversation_id": "conversation-1",
                "conversation_title": "Unified Deep Agent",
                "channel_key": "feishu:test",
                "created_at": "2026-09-02T00:00:00Z",
                "last_active_at": "2026-09-02T00:00:00Z",
                "title_generated": False,
            },
            as_node="__start__",
        )

        saved = await worker.aget_state(config)
        self.assertEqual(saved.values["conversation_id"], "conversation-1")
        self.assertEqual(
            saved.values["conversation_title"],
            "Unified Deep Agent",
        )


if __name__ == "__main__":
    unittest.main()
