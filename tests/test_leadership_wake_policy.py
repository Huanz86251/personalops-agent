"""Provider-free tests for leadership wake coalescing rules."""

from datetime import datetime, timezone
import unittest

from workers.leadership_models import LeadershipDecision
from workers.leadership import WorkerLeadershipBridge
from workers.leadership_models import LeadershipDecisionResult
from eventing import AsyncEventStore
from pathlib import Path
from tempfile import TemporaryDirectory
from workers.progress import WorkerProgressPayload, WorkerProgressRecord
from workers.wake_policy import LeadershipWakePolicy


def report(worker_id: str, sequence: int, claim: str = "not_ready"):
    return WorkerProgressRecord(
        sequence=sequence,
        total_tool_calls=sequence * 5,
        published_at=datetime.now(timezone.utc),
        worker_id=worker_id,
        event_id="event-1",
        step_id="1",
        progress=WorkerProgressPayload(
            phase="working",
            summary=f"checkpoint {sequence}",
            next_action="continue",
            completion_claim=claim,
        ),
    )


class LeadershipWakePolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = LeadershipWakePolicy(
            single_worker_reports=2,
            multi_worker_reports=1,
        )

    def test_single_worker_wakes_on_second_unread_report(self):
        self.assertIsNone(
            self.policy.evaluate(
                active_worker_ids=["worker-a"],
                reports_by_worker={"worker-a": [report("worker-a", 1)]},
            )
        )
        self.assertEqual(
            self.policy.evaluate(
                active_worker_ids=["worker-a"],
                reports_by_worker={
                    "worker-a": [report("worker-a", 1), report("worker-a", 2)]
                },
            ),
            "SINGLE_WORKER_THRESHOLD",
        )

    def test_multi_worker_uses_non_blocking_watermark_barrier(self):
        self.assertIsNone(
            self.policy.evaluate(
                active_worker_ids=["worker-a", "worker-b"],
                reports_by_worker={"worker-a": [report("worker-a", 1)]},
            )
        )
        self.assertEqual(
            self.policy.evaluate(
                active_worker_ids=["worker-a", "worker-b"],
                reports_by_worker={
                    "worker-a": [report("worker-a", 1)],
                    "worker-b": [report("worker-b", 1)],
                },
            ),
            "MULTI_WORKER_BARRIER",
        )

    def test_blocked_and_ready_claims_bypass_thresholds(self):
        self.assertEqual(
            self.policy.evaluate(
                active_worker_ids=["worker-a", "worker-b"],
                reports_by_worker={
                    "worker-a": [report("worker-a", 1, "blocked")]
                },
            ),
            "WORKER_BLOCKED",
        )
        self.assertEqual(
            self.policy.evaluate(
                active_worker_ids=["worker-a"],
                reports_by_worker={
                    "worker-a": [report("worker-a", 1, "possibly_ready")]
                },
            ),
            "WORKER_POSSIBLY_READY",
        )

    def test_action_specific_payload_is_validated(self):
        with self.assertRaisesRegex(ValueError, "GUIDE requires guidance"):
            LeadershipDecision(action="GUIDE", reason="redirect")
        with self.assertRaisesRegex(
            ValueError,
            "REPLACE requires replacement_assignment",
        ):
            LeadershipDecision(action="REPLACE", reason="restart")


class LeadershipBarrierCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_possibly_ready_bypasses_scheduler_and_accepts_for_review(self):
        captured = []

        async def decide(request):
            captured.append(request)
            raise AssertionError("possibly_ready must not invoke Scheduler")

        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                bridge = WorkerLeadershipBridge(
                    worker_graph=object(),
                    event_store=store,
                    wake_policy=LeadershipWakePolicy(
                        single_worker_reports=2,
                        multi_worker_reports=1,
                    ),
                    decision_handler=decide,
                )
                await bridge.register_worker(
                    event_id="event-1",
                    worker_id="worker-a",
                    step_id="1",
                    assignment="find sufficient evidence",
                )
                ready = report("worker-a", 1, "possibly_ready")
                await store.append_worker_progress(ready.model_dump(mode="json"))

                result = await bridge.decide_at_progress_gate(ready)

                self.assertEqual(result.decision.action, "ACCEPT")
                self.assertEqual(result.model_rounds_used, 0)
                self.assertEqual(captured, [])
            finally:
                await store.close()

    async def test_two_workers_trigger_one_coalesced_leader_call(self):
        captured = []

        async def decide(request):
            captured.append(request)
            return LeadershipDecisionResult(
                decision=LeadershipDecision(
                    action="CONTINUE",
                    reason="Both workers are making healthy progress.",
                ),
                model_rounds_used=1,
            )

        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                bridge = WorkerLeadershipBridge(
                    worker_graph=object(),
                    event_store=store,
                    wake_policy=LeadershipWakePolicy(
                        single_worker_reports=2,
                        multi_worker_reports=1,
                    ),
                    decision_handler=decide,
                )
                for worker_id in ("worker-a", "worker-b"):
                    await bridge.register_worker(
                        event_id="event-1",
                        worker_id=worker_id,
                        step_id="1",
                        assignment=f"assignment for {worker_id}",
                    )

                first = report("worker-a", 1)
                await store.append_worker_progress(first.model_dump(mode="json"))
                first_result = await bridge.decide_at_progress_gate(first)
                self.assertEqual(first_result.model_rounds_used, 0)
                self.assertEqual(captured, [])

                second = report("worker-b", 1)
                await store.append_worker_progress(second.model_dump(mode="json"))
                second_result = await bridge.decide_at_progress_gate(second)
                self.assertEqual(second_result.model_rounds_used, 1)
                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0].reason, "MULTI_WORKER_BARRIER")
                self.assertEqual(len(captured[0].workers), 2)
                self.assertEqual(
                    await store.get_worker_progress_cursor(
                        consumer_id="planning_supervisor",
                        worker_id="worker-a",
                    ),
                    1,
                )
                self.assertEqual(
                    await store.get_worker_progress_cursor(
                        consumer_id="planning_supervisor",
                        worker_id="worker-b",
                    ),
                    1,
                )
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
