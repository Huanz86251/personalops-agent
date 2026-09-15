import ast
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock
from tempfile import TemporaryDirectory

from eventing import EventAction, EventOrigin, create_agent_event, create_event_run
from eventing.store import AsyncEventStore
from scheduling import DeliveryTarget, ReminderRun, ReminderSchedule
from scheduling.models import Recurrence, create_schedule


def delivery_namespace(event_store, pump, send_text):
    source = Path(__file__).resolve().parents[1] / "main.py"
    wanted = {"_deliver_scheduled_feishu_text", "_enqueue_scheduled_agent_event"}
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    selected = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert len(selected) == len(wanted)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    namespace = {
        "Any": object,
        "ReminderRun": ReminderRun,
        "ReminderSchedule": ReminderSchedule,
        "EventAction": EventAction,
        "EventOrigin": EventOrigin,
        "create_agent_event": create_agent_event,
        "create_event_run": create_event_run,
        "conversation_runtime": SimpleNamespace(event_store=event_store),
        "FEISHU_EVENT_CONTEXTS": {},
        "_require_event_pump": lambda: pump,
        "_build_progress_callback": lambda chat: AsyncMock(),
        "_send_text": send_text,
        "logger": logging.getLogger("scheduled-delivery-test"),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


class ScheduledDeliveryTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.events = AsyncEventStore(Path(self.temporary.name) / "events.sqlite3")
        await self.events.start()
        self.addAsyncCleanup(self.events.close)
        self.pump = SimpleNamespace(notify=Mock())
        self.send_text = AsyncMock()
        self.ns = delivery_namespace(self.events, self.pump, self.send_text)

    def schedule(self, target):
        return create_schedule(
            title="定时测试",
            message="查询上海天气并总结",
            run_at=datetime.now(timezone.utc),
            timezone_name="Asia/Shanghai",
            recurrence=Recurrence.ONCE,
            delivery_target=target,
            reply_target_id="chat-trusted",
            conversation_id="conv-trusted",
        )

    async def test_feishu_text_uses_persisted_recipient(self):
        schedule = self.schedule(DeliveryTarget.FEISHU)
        run = SimpleNamespace(run_id="srun_text")
        receipt = await self.ns["_deliver_scheduled_feishu_text"](schedule, run)
        self.send_text.assert_awaited_once_with("chat-trusted", schedule.message)
        self.assertEqual(receipt["provider"], "feishu")

    async def test_agent_event_is_queue_system_and_idempotent(self):
        schedule = self.schedule(DeliveryTarget.AGENT_EVENT)
        run = SimpleNamespace(run_id="srun_fixed")
        first = await self.ns["_enqueue_scheduled_agent_event"](schedule, run)
        second = await self.ns["_enqueue_scheduled_agent_event"](schedule, run)

        self.assertEqual(first["event_id"], "evt_schedule_fixed")
        self.assertEqual(second["event_id"], first["event_id"])
        event = await self.events.require_event(first["event_id"])
        self.assertIs(event.action, EventAction.QUEUE)
        self.assertIs(event.origin, EventOrigin.SYSTEM)
        self.assertEqual(event.reply_target_id, "chat-trusted")
        self.assertIsNotNone(await self.events.get_run(event.event_id))
        self.assertEqual(self.pump.notify.call_count, 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
