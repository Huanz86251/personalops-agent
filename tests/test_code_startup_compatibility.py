"""Offline startup checks with persisted Events and incompatible CODE history."""

from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

os.environ["PHOENIX_TRACING_ENABLED"] = "false"

from conversation_runtime import ConversationRuntime
from eventing import (
    AsyncEventStore,
    EventAction,
    EventOrigin,
    EventStatus,
    RunStatus,
    create_agent_event,
    create_event_run,
)
from run_workspace import initialize_run_workspace
from workers.code_runtime import CodeStepRuntime


async def seed_run(store, event_id, status):
    event = replace(
        create_agent_event(
            event_id=event_id,
            conversation_id="offline-conversation",
            action=EventAction.QUEUE,
            payload_text="Offline task.",
            origin=EventOrigin.SYSTEM,
        ),
        status=EventStatus.APPLIED,
    )
    await store.add_event(event, run=replace(create_event_run(event), status=status))


async def seed_control(store, target, action):
    event = replace(
        create_agent_event(
            event_id=f"control-{target}",
            conversation_id="offline-conversation",
            action=action,
            payload_text="" if action is EventAction.CANCEL else "Replacement task.",
            target_event_id=target,
            origin=EventOrigin.SYSTEM,
        ),
        status=EventStatus.APPLIED,
    )
    await store.add_event(event)


def write_legacy_archive(root, run_id):
    archive = root / "archives" / run_id
    archive.mkdir(parents=True)
    path = archive / "runtime_checkpoint.json"
    original = json.dumps(
        {
            "run_id": run_id,
            "phase": "RECOVERY_REQUIRED",
            "recovery_state": {
                "contract": {"implementation_guidance": ["Historical hint."]},
                "review_loop": {
                    "scheduler_epoch": 1,
                    "pending_scheduler_directive": None,
                    "scheduler_continuations": [],
                    "pending_instruction": {"findings": [{"category": "old"}]},
                },
            },
        }
    ).encode("utf-8")
    path.write_bytes(original)
    return path, original


def code_runtime(root):
    return CodeStepRuntime(
        object(),
        sandbox_manager=Mock(),
        source_root=root / "workspace",
        target_root=root / "workspace",
        archive_root=root / "archives",
        run_storage_root=root / "runs",
    )


class CodeStartupCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_history_is_filtered_before_schema_and_unrelated_cancel_succeeds(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = AsyncEventStore(root / "events.sqlite3")
            await store.start()
            try:
                records = []
                for status in (
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                    RunStatus.SUPERSEDED,
                ):
                    run_id = f"old-{status.value.lower()}"
                    await seed_run(store, run_id, status)
                    records.append(write_legacy_archive(root, run_id))
                await seed_run(store, "unrelated-cancelled", RunStatus.CANCELLED)
                await seed_control(store, "unrelated-cancelled", EventAction.CANCEL)
                conversation = ConversationRuntime.__new__(ConversationRuntime)
                conversation.event_store = store
                runtime = code_runtime(root)
                with patch(
                    "conversation_runtime.initialize_run_workspace",
                    side_effect=lambda run_id: initialize_run_workspace(
                        run_id, storage_root=root / "runs"
                    ),
                ):
                    terminal_runs = await conversation._recover_code_startup(runtime)

                self.assertEqual(len(terminal_runs), 5)
                self.assertFalse(runtime.recovered_session_ids)
                self.assertFalse(runtime.startup_recovery_blockers)
                self.assertFalse(runtime.sandbox_manager.mock_calls)
                layout = initialize_run_workspace(
                    "unrelated-cancelled", storage_root=root / "runs"
                )
                receipts = list(layout.receipts_root.glob("*.json"))
                self.assertEqual(len(receipts), 1)
                self.assertEqual(
                    json.loads(receipts[0].read_text(encoding="utf-8"))["kind"],
                    "RUN_CANCELLED",
                )
                for path, original in records:
                    self.assertEqual(path.read_bytes(), original)
            finally:
                await store.close()

    async def test_owned_incompatible_control_preserves_evidence_and_does_not_abort_startup(self):
        for status, action in (
            (RunStatus.CANCELLED, EventAction.CANCEL),
            (RunStatus.SUPERSEDED, EventAction.REPLACE),
        ):
            with self.subTest(action=action), TemporaryDirectory() as temporary:
                root = Path(temporary)
                store = AsyncEventStore(root / "events.sqlite3")
                await store.start()
                try:
                    await seed_run(store, "owned-old-run", status)
                    await seed_control(store, "owned-old-run", action)
                    path, original = write_legacy_archive(root, "owned-old-run")
                    conversation = ConversationRuntime.__new__(ConversationRuntime)
                    conversation.event_store = store
                    runtime = code_runtime(root)
                    with (
                        patch("conversation_runtime.initialize_run_workspace") as layout,
                        patch("conversation_runtime.write_run_cancellation_receipt") as cancel,
                        patch("conversation_runtime.write_run_supersession_receipt") as supersede,
                        patch("conversation_runtime.inherit_replacement_workspace") as inherit,
                        self.assertLogs("agent", level="WARNING") as logs,
                    ):
                        await conversation._recover_code_startup(runtime)
                    self.assertIn("未完成", logs.output[-1])
                    layout.assert_not_called()
                    cancel.assert_not_called()
                    supersede.assert_not_called()
                    inherit.assert_not_called()
                    self.assertFalse(runtime.sandbox_manager.mock_calls)
                    self.assertEqual(path.read_bytes(), original)
                finally:
                    await store.close()

    async def test_incompatible_active_run_is_blocked_instead_of_silently_accepted(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = AsyncEventStore(root / "events.sqlite3")
            await store.start()
            try:
                await seed_run(store, "active-old-run", RunStatus.PAUSED)
                path, original = write_legacy_archive(root, "active-old-run")
                conversation = ConversationRuntime.__new__(ConversationRuntime)
                conversation.event_store = store
                runtime = code_runtime(root)
                await conversation._recover_code_startup(runtime)
                self.assertEqual(len(runtime.startup_recovery_blockers), 1)
                self.assertFalse(runtime.recovered_session_ids)
                self.assertFalse(runtime.sandbox_manager.mock_calls)
                self.assertEqual(path.read_bytes(), original)
            finally:
                await store.close()
