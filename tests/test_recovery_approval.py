import asyncio
import unittest

from recovery_approval import RecoveryApprovalRequest, RecoveryApprovalService


def sample_request() -> RecoveryApprovalRequest:
    return RecoveryApprovalRequest(
        event_id="evt-1",
        chat_id="chat-1",
        step_id=2,
        worker_kind="WEB",
        original_user_request="Find the most popular movie tonight.",
        current_step='{"step_id":2}',
        downgrade_reason="No popularity ranking was verified.",
        proposed_degraded_outcome="Return one verified available movie.",
        user_confirmation_request="May I return one verified available movie instead?",
    )


class RecoveryApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def test_allow_resumes_and_is_cached(self):
        service = RecoveryApprovalService()
        notified = asyncio.Event()

        async def notify(_chat_id, _request):
            notified.set()

        service.configure_human_notifier(notify)
        task = asyncio.create_task(service.request_decision(sample_request()))
        await notified.wait()
        reply = service.consume_user_reply("chat-1", "允许")
        decision = await task

        self.assertTrue(reply.handled)
        self.assertEqual(decision.status, "APPROVED")
        cached = await service.request_decision(sample_request())
        self.assertEqual(cached.status, "APPROVED")

    async def test_deny_is_explicit_and_cached(self):
        service = RecoveryApprovalService()
        notified = asyncio.Event()

        async def notify(_chat_id, _request):
            notified.set()

        service.configure_human_notifier(notify)
        task = asyncio.create_task(service.request_decision(sample_request()))
        await notified.wait()
        reply = service.consume_user_reply("chat-1", "拒绝")
        decision = await task

        self.assertTrue(reply.handled)
        self.assertEqual(decision.status, "DENIED")
        self.assertIn("拒绝", decision.message)
        cached = await service.request_decision(sample_request())
        self.assertEqual(cached.status, "DENIED")

    async def test_unknown_reply_keeps_request_pending(self):
        service = RecoveryApprovalService()
        notified = asyncio.Event()

        async def notify(_chat_id, _request):
            notified.set()

        service.configure_human_notifier(notify)
        task = asyncio.create_task(service.request_decision(sample_request()))
        await notified.wait()
        reply = service.consume_user_reply("chat-1", "我考虑一下")
        self.assertTrue(reply.handled)
        self.assertIn("只回复", reply.reply)
        self.assertFalse(task.done())
        service.consume_user_reply("chat-1", "允许")
        self.assertEqual((await task).status, "APPROVED")

    async def test_unavailable_channel_never_approves(self):
        service = RecoveryApprovalService()
        request = sample_request().model_copy(update={"chat_id": ""})
        decision = await service.request_decision(request)
        self.assertEqual(decision.status, "UNAVAILABLE")
        self.assertEqual(decision.source, "HARNESS")


if __name__ == "__main__":
    unittest.main()
