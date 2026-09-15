import base64
import sqlite3
from datetime import datetime, timedelta, timezone
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.utils.function_calling import convert_to_openai_tool

from scheduling import (
    AsyncScheduleStore,
    DeliveryTarget,
    NotificationReceipt,
    Recurrence,
    ReminderRunStatus,
    ScheduleService,
    ScheduleStatus,
    WindowsToastNotifier,
)
from scheduling.models import next_occurrence, parse_run_at
from tools import ALL_TOOLS
from tools.schedule_tools import (
    configure_schedule_service,
    schedule_create,
    schedule_create_agent_task,
    schedule_create_feishu_reminder,
)
from toolsets import DEFAULT_TOOLSET_REGISTRY


class FakeNotifier:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def send(self, **payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return NotificationReceipt(True, "fake", "accepted")


class ScheduleModelTests(unittest.TestCase):
    def test_run_at_requires_explicit_timezone(self):
        with self.assertRaisesRegex(ValueError, "时区"):
            parse_run_at("2026-09-07T09:00:00")
        parsed = parse_run_at("2026-09-07T09:00:00+08:00")
        self.assertEqual(parsed.hour, 1)

    def test_recurrence_keeps_local_clock_and_skips_backlog(self):
        first = parse_run_at("2026-09-01T09:00:00+08:00")
        after = parse_run_at("2026-09-05T10:00:00+08:00")
        following = next_occurrence(
            first, Recurrence.DAILY, "Asia/Shanghai", after=after
        )
        self.assertEqual(
            following,
            parse_run_at("2026-09-06T09:00:00+08:00"),
        )

    def test_toast_script_base64_encodes_untrusted_text(self):
        hostile = "</text><script>bad</script>;$env:SECRET"
        encoded = WindowsToastNotifier._encoded_script(
            app_id="PersonalOps.Agent",
            title=hostile,
            message=hostile,
            tag="tag",
        )
        script = base64.b64decode(encoded).decode("utf-16le")
        self.assertNotIn(hostile, script)
        self.assertIn("SecurityElement", script)


class ScheduleStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.store = AsyncScheduleStore(Path(self.temporary.name) / "schedules.sqlite3")
        await self.store.start()

    async def asyncTearDown(self):
        await self.store.close()
        self.temporary.cleanup()

    async def test_once_schedule_materializes_and_completes(self):
        due = datetime.now(timezone.utc) - timedelta(seconds=1)
        schedule = await self.store.create(
            title="测试",
            message="该做事了",
            run_at=due,
            timezone_name="Asia/Shanghai",
            recurrence=Recurrence.ONCE,
        )
        self.assertEqual(await self.store.materialize_due(), 1)
        persisted = await self.store.get(schedule.schedule_id)
        self.assertEqual(persisted.status, ScheduleStatus.COMPLETED)
        claimed_schedule, run = await self.store.claim_next()
        self.assertEqual(claimed_schedule.schedule_id, schedule.schedule_id)
        self.assertEqual(run.status, ReminderRunStatus.CLAIMED)
        await self.store.finish_run(run.run_id, ReminderRunStatus.SUBMITTED)
        runs = await self.store.list_runs(schedule.schedule_id)
        self.assertEqual(runs[0].status, ReminderRunStatus.SUBMITTED)

    async def test_pause_resume_and_soft_delete(self):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        schedule = await self.store.create(
            title="周报",
            message="提交周报",
            run_at=future,
            timezone_name="Asia/Shanghai",
            recurrence=Recurrence.WEEKLY,
        )
        paused = await self.store.set_status(schedule.schedule_id, ScheduleStatus.PAUSED)
        self.assertEqual(paused.status, ScheduleStatus.PAUSED)
        resumed = await self.store.set_status(schedule.schedule_id, ScheduleStatus.ACTIVE)
        self.assertEqual(resumed.status, ScheduleStatus.ACTIVE)
        cancelled = await self.store.set_status(
            schedule.schedule_id, ScheduleStatus.CANCELLED
        )
        self.assertEqual(cancelled.status, ScheduleStatus.CANCELLED)
        with self.assertRaisesRegex(ValueError, "不能"):
            await self.store.set_status(schedule.schedule_id, ScheduleStatus.ACTIVE)

    async def test_runner_records_submission_without_model(self):
        notifier = FakeNotifier()
        service = ScheduleService(self.store, notifier, poll_seconds=3600)
        # The test owns the already-started store and invokes one deterministic scan.
        await self.store.create(
            title="本地提醒",
            message="正文",
            run_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            timezone_name="Asia/Shanghai",
            recurrence=Recurrence.ONCE,
        )
        self.assertEqual(await service.run_once(), 1)
        self.assertEqual(notifier.calls[0]["title"], "本地提醒")

    async def test_runner_dispatches_by_persisted_target(self):
        delivered = []

        async def deliver_feishu(schedule, run):
            delivered.append((schedule.delivery_target, schedule.message, run.run_id))
            return {"provider": "fake-feishu"}

        service = ScheduleService(self.store, FakeNotifier(), poll_seconds=3600)
        service.configure_delivery_handler(DeliveryTarget.FEISHU, deliver_feishu)
        await self.store.create(
            title="飞书提醒",
            message="提交材料",
            run_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            timezone_name="Asia/Shanghai",
            recurrence=Recurrence.ONCE,
            delivery_target=DeliveryTarget.FEISHU,
            reply_target_id="chat-trusted",
            conversation_id="conv-trusted",
        )
        self.assertEqual(await service.run_once(), 1)
        self.assertEqual(delivered[0][:2], (DeliveryTarget.FEISHU, "提交材料"))


class ScheduleMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_windows_database_is_upgraded_in_place(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                """CREATE TABLE reminder_schedules (
                    schedule_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    message TEXT NOT NULL, recurrence TEXT NOT NULL,
                    timezone_name TEXT NOT NULL, next_run_at TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            now = "2099-01-01T01:00:00Z"
            connection.execute(
                "INSERT INTO reminder_schedules VALUES (?,?,?,?,?,?,?,?,?)",
                ("sch_old", "旧提醒", "正文", "ONCE", "Asia/Shanghai", now,
                 "ACTIVE", now, now),
            )
            connection.commit()
            connection.close()

            store = AsyncScheduleStore(path)
            await store.start()
            try:
                restored = await store.get("sch_old")
                self.assertEqual(restored.delivery_target, DeliveryTarget.WINDOWS)
                self.assertIsNone(restored.reply_target_id)
            finally:
                await store.close()


class ScheduleToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_have_precise_schema_and_toolset_card(self):
        names = {tool.name for tool in ALL_TOOLS}
        expected = {
            "schedule_create",
            "schedule_create_feishu_reminder",
            "schedule_create_agent_task",
            "schedule_list",
            "schedule_pause",
            "schedule_resume",
            "schedule_delete",
            "schedule_runs",
            "windows_notify",
        }
        self.assertTrue(expected <= names)
        schema = convert_to_openai_tool(schedule_create)["function"]
        run_at = schema["parameters"]["properties"]["run_at"]
        self.assertIn("ISO-8601", run_at["description"])
        self.assertIn("时区", run_at["description"])
        feishu_schema = convert_to_openai_tool(schedule_create_feishu_reminder)["function"]
        agent_schema = convert_to_openai_tool(schedule_create_agent_task)["function"]
        self.assertNotIn("runtime", feishu_schema["parameters"]["properties"])
        self.assertNotIn("chat_id", feishu_schema["parameters"]["properties"])
        self.assertNotIn("runtime", agent_schema["parameters"]["properties"])
        resolution = DEFAULT_TOOLSET_REGISTRY.resolve("SCHEDULED_AUTOMATION", ALL_TOOLS)
        self.assertTrue(resolution.is_available)
        self.assertTrue(expected <= set(resolution.tool_names))
        self.assertIn("明天上午九点提醒我", resolution.spec.routing_profile)

    async def test_create_tool_persists_and_returns_id(self):
        with TemporaryDirectory() as temporary:
            store = AsyncScheduleStore(Path(temporary) / "tool.sqlite3")
            service = ScheduleService(store, FakeNotifier(), poll_seconds=3600)
            await service.start()
            configure_schedule_service(service)
            try:
                result = await schedule_create.ainvoke(
                    {
                        "title": "明日事项",
                        "message": "回复邮件",
                        "run_at": "2099-01-02T09:00:00+08:00",
                        "timezone_name": "Asia/Shanghai",
                        "recurrence": "ONCE",
                    }
                )
                self.assertTrue(result["schedule_id"].startswith("sch_"))
                self.assertEqual(result["status"], "ACTIVE")
            finally:
                configure_schedule_service(None)
                await service.stop()


if __name__ == "__main__":
    unittest.main()
