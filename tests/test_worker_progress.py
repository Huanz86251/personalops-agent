"""Offline tests for generic Web and Code worker progress reporting."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.tools import (
    tool,
)
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import (
    Field,
)

from eventing import AsyncEventStore

from workers.web_worker import (
    create_web_worker,
)
from workers.progress import (
    PUBLISH_WORKER_PROGRESS_NAME,
    WorkerProgressMiddleware,
    count_tool_calls_since_progress,
)
from workers.leadership import WorkerLeadershipBridge
from workers.leadership_models import (
    LeadershipDecision,
    LeadershipDecisionResult,
)
from workers.wake_policy import LeadershipWakePolicy


class FinalizationToolRoutingTests(unittest.TestCase):
    def test_code_finalization_does_not_require_unavailable_web_submit_tool(self):
        names = ("submit_code_for_review", "respond_to_code_review", "submit_continued_code_for_review")
        middleware = WorkerProgressMiddleware(finalization_tools=names)
        request = SimpleNamespace(state={"worker_finalize_requested": True},
            tools=[SimpleNamespace(name=name) for name in (*names, "execute")],
            override=lambda **changes: changes)
        result = middleware._require_control_request(request)
        self.assertEqual([tool.name for tool in result["tools"]], list(names))
        self.assertEqual(result["tool_choice"], "required")


@tool
def web_search(
    value: str,
) -> str:
    """Return a value so progress tests can make a real tool call."""

    return value


class ProgressScriptModel(
    BaseChatModel
):
    """Provider-free model that follows the forced progress checkpoint."""

    invocation_count: int = 0
    current_tool_names: list[str] = Field(
        default_factory=list
    )
    bound_tool_history: list[
        list[str]
    ] = Field(
        default_factory=list
    )
    system_prompt_history: list[str] = Field(
        default_factory=list
    )

    @property
    def _llm_type(
        self,
    ) -> str:
        return (
            "worker-progress-script-test"
        )

    def bind_tools(
        self,
        tools,
        **kwargs,
    ):
        self.current_tool_names = [
            current_tool.name
            for current_tool
            in tools
        ]
        self.bound_tool_history.append(
            list(
                self.current_tool_names
            )
        )
        return self

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        system_text = "\n".join(
            str(message.content)
            for message
            in messages
            if isinstance(
                message,
                SystemMessage,
            )
        )
        self.system_prompt_history.append(
            system_text
        )
        invocation_index = (
            self.invocation_count
        )
        self.invocation_count += 1

        if invocation_index < 4:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": (
                            "web_search"
                        ),
                        "args": {
                            "value": (
                                f"probe-{invocation_index}"
                            )
                        },
                        "id": (
                            f"probe-call-{invocation_index}"
                        ),
                        "type": "tool_call",
                    }
                ],
            )

        elif invocation_index == 4:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": (
                            PUBLISH_WORKER_PROGRESS_NAME
                        ),
                        "args": {
                            "progress": {
                                "phase": "researching",
                                "summary": (
                                    "Completed four tool calls "
                                    "and collected initial evidence."
                                ),
                                "findings": [
                                    "The probe returned four results."
                                ],
                                "difficulties": [],
                                "evidence_refs": [
                                    "artifact://probe-results"
                                ],
                                "next_action": (
                                    "Continue with the assigned task."
                                ),
                                "completion_claim": (
                                    "not_ready"
                                ),
                            }
                        },
                        "id": "progress-call-1",
                        "type": "tool_call",
                    }
                ],
            )

        else:
            message = AIMessage(
                content=(
                    "Worker continued after publishing progress."
                )
            )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message
                )
            ]
        )


class WorkerProgressTests(
    unittest.TestCase
):
    def test_every_tool_call_is_counted_and_checkpoint_resets_interval(
        self,
    ) -> None:
        model = ProgressScriptModel()
        worker = create_web_worker(
            model,
            tools=[
                web_search
            ],
            progress_every_tool_calls=4,
        )

        result = worker.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Run until the progress checkpoint."
                        ),
                    }
                ],
                "worker_id": "worker-test-01",
                "event_id": "event-test-01",
                "step_id": "step-test-01",
                "skill_mode": "off",
            }
        )

        self.assertEqual(
            result["messages"][-1].content,
            "Worker continued after publishing progress.",
        )
        self.assertEqual(
            result[
                "worker_total_tool_calls"
            ],
            5,
        )
        self.assertEqual(
            result[
                "worker_last_progress_tool_call"
            ],
            5,
        )
        self.assertEqual(
            count_tool_calls_since_progress(
                result
            ),
            0,
        )

        reports = result[
            "worker_progress_reports"
        ]
        self.assertEqual(
            len(reports),
            1,
        )
        report = reports[0]
        self.assertEqual(
            report["sequence"],
            1,
        )
        self.assertEqual(
            report["total_tool_calls"],
            5,
        )
        self.assertEqual(
            report["worker_id"],
            "worker-test-01",
        )
        self.assertEqual(
            report["event_id"],
            "event-test-01",
        )
        self.assertEqual(
            report["step_id"],
            "step-test-01",
        )

        forced_tool_sets = [
            tool_names
            for tool_names
            in model.bound_tool_history
            if tool_names
            == [
                PUBLISH_WORKER_PROGRESS_NAME
            ]
        ]
        self.assertEqual(
            len(forced_tool_sets),
            1,
        )
        self.assertEqual(model.system_prompt_history[4], model.system_prompt_history[5])
        self.assertTrue(any(
            getattr(message, "additional_kwargs", {}).get("personalops_runtime_event")
            and "publish_worker_progress" in str(message.content)
            for message in result["messages"]
        ))
        self.assertEqual(
            model.bound_tool_history[-1],
            ["submit_for_review"],
        )
        self.assertEqual(result["worker_finalize_reason"], "NATURAL_EXIT")
        self.assertEqual(result["worker_finalization_model_calls_used"], 1)

    def test_interval_is_bounded(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            WorkerProgressMiddleware(
                every_tool_calls=1
            )

        with self.assertRaises(
            ValueError
        ):
            WorkerProgressMiddleware(
                every_tool_calls=9
            )


class AsyncWorkerProgressTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_async_worker_uses_same_forced_checkpoint(
        self,
    ) -> None:
        model = ProgressScriptModel()
        worker = create_web_worker(
            model,
            tools=[
                web_search
            ],
            progress_every_tool_calls=4,
        )

        result = await worker.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Run the asynchronous progress flow."
                        ),
                    }
                ],
                "worker_id": "async-worker-test-01",
                "skill_mode": "off",
            }
        )

        self.assertEqual(
            result[
                "worker_total_tool_calls"
            ],
            5,
        )
        self.assertEqual(
            count_tool_calls_since_progress(
                result
            ),
            0,
        )
        self.assertEqual(
            len(
                result[
                    "worker_progress_reports"
                ]
            ),
            1,
        )
        self.assertIn(
            [
                PUBLISH_WORKER_PROGRESS_NAME
            ],
            model.bound_tool_history,
        )

    async def test_leadership_bridge_persists_real_custom_stream_and_cursor(
        self,
    ) -> None:
        model = ProgressScriptModel()
        worker = create_web_worker(
            model,
            tools=[web_search],
            progress_every_tool_calls=4,
        )

        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                bridge = WorkerLeadershipBridge(worker, store)
                result = await bridge.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Run through leadership.",
                            }
                        ],
                        "worker_id": "worker-leadership-01",
                        "event_id": "event-leadership-01",
                        "step_id": "1",
                        "skill_mode": "off",
                    }
                )

                self.assertEqual(result["worker_total_tool_calls"], 5)
                event_reports = await bridge.read_event_progress(
                    "event-leadership-01"
                )
                self.assertEqual(len(event_reports), 1)
                self.assertEqual(
                    event_reports[0].worker_id,
                    "worker-leadership-01",
                )

                batch = await bridge.read_pending_progress(
                    worker_id="worker-leadership-01"
                )
                self.assertEqual(batch.cursor_before, 0)
                self.assertEqual(batch.latest_sequence, 1)
                self.assertEqual(len(batch.reports), 1)

                await bridge.acknowledge_progress(batch)
                empty_batch = await bridge.read_pending_progress(
                    worker_id="worker-leadership-01"
                )
                self.assertEqual(empty_batch.cursor_before, 1)
                self.assertEqual(empty_batch.reports, [])
            finally:
                await store.close()

    async def test_progress_interrupt_resumes_with_leader_accept(self) -> None:
        model = ProgressScriptModel()
        worker = create_web_worker(
            model,
            tools=[web_search],
            progress_every_tool_calls=4,
            checkpointer=InMemorySaver(),
        )
        captured_wakes = []

        async def accept(request):
            captured_wakes.append(request)
            return LeadershipDecisionResult(
                decision=LeadershipDecision(
                    action="ACCEPT",
                    reason="The progress evidence is sufficient.",
                ),
                model_rounds_used=1,
            )

        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                bridge = WorkerLeadershipBridge(
                    worker,
                    store,
                    wake_policy=LeadershipWakePolicy(
                        single_worker_reports=1,
                        multi_worker_reports=1,
                    ),
                    decision_handler=accept,
                )
                result = await bridge.ainvoke(
                    {
                        "messages": [
                            {"role": "user", "content": "Reach one safe point."}
                        ],
                        "worker_id": "worker-accept-01",
                        "event_id": "event-accept-01",
                        "step_id": "1",
                        "skill_mode": "off",
                    },
                    config={
                        "configurable": {"thread_id": "worker-accept-thread"}
                    },
                )

                self.assertEqual(len(captured_wakes), 1)
                self.assertEqual(result["worker_terminal_action"], "ACCEPT")
                self.assertEqual(
                    result["worker_leadership_model_rounds_used"],
                    1,
                )
                self.assertEqual(
                    result["worker_leadership_decisions"][-1]["decision"]["action"],
                    "ACCEPT",
                )
                self.assertIsNone(
                    await store.get_pending_worker_directive(
                        "worker-accept-01"
                    )
                )
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
