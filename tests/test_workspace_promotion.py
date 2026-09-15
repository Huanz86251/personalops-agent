from __future__ import annotations

import tempfile
import unittest
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from delivery.models import (
    DeliveryApprovalMode,
    PromotionStatus,
    transition_workspace_promotion,
)
from delivery.service import WorkspacePromotionService
from artifact_models import ResolvedArtifactCandidate
from artifact_publisher import publish_artifact_to_handoff
from eventing.store import AsyncEventStore
from integration_repository import IntegrationRepository
from planning_models import StepArtifactOutput
from run_workspace import (
    cleanup_expired_run_workspaces,
    initialize_code_integration,
    initialize_run_workspace,
)


class WorkspacePromotionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = AsyncEventStore(self.root / "events.sqlite3")
        await self.store.start()

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.temporary.cleanup()

    def accepted_layout(self, run_id: str = "evt-code"):
        source = self.root / "source"
        source.mkdir(exist_ok=True)
        (source / "app.py").write_text("print('old')\n", encoding="utf-8")
        layout = initialize_run_workspace(
            run_id,
            storage_root=self.root / "runs",
            canonical_root=self.root / "legacy-workspace",
        )
        integration = initialize_code_integration(layout=layout, source_root=source)
        repository = IntegrationRepository(
            run_id=run_id,
            root=integration,
            receipts_root=layout.receipts_root,
        )
        repository.initialize()
        (integration / "app.py").write_text("print('reviewed')\n", encoding="utf-8")
        repository.commit_accepted(
            step_id=1,
            candidate_revision=1,
            manifest_id="manifest-one",
            publication_id="publication-one",
            approved_paths=("app.py",),
            review_summary="Reviewer approved the change.",
            test_summary="unit=PASSED",
        )
        return layout

    def service(self, mode: DeliveryApprovalMode) -> WorkspacePromotionService:
        return WorkspacePromotionService(
            event_store=self.store,
            approval_mode=mode,
            conversation_workspace_root=self.root / "workspaces",
            metadata_root=self.root / "workspace-metadata",
        )

    def add_web_artifact(
        self,
        layout,
        *,
        disposition: str = "USER_DELIVERABLE",
    ):
        source = self.root / f"download-{layout.run_id}.html"
        source.write_text("<main>reviewed web artifact</main>", encoding="utf-8")
        payload = source.read_bytes()
        candidate = ResolvedArtifactCandidate(
            candidate_id="downloaded-template",
            output_id="requested_html",
            kind="DOWNLOADED_FILE",
            description="HTML requested by the user.",
            verified=True,
            location=str(source),
            storage_path=str(source),
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        output = StepArtifactOutput(
            output_id="requested_html",
            description="HTML requested by the user.",
            disposition=disposition,
            target_path=(
                "downloads/template.html"
                if disposition == "USER_DELIVERABLE"
                else None
            ),
        )
        return publish_artifact_to_handoff(
            layout=layout,
            candidate=candidate,
            output=output,
        )

    async def test_human_approval_is_separate_from_reviewer_application(self) -> None:
        layout = self.accepted_layout()
        service = self.service(DeliveryApprovalMode.HUMAN)
        promotion = await service.prepare_from_run(
            layout=layout,
            conversation_id="conv-one",
            final_status="COMPLETED",
        )
        self.assertIsNotNone(promotion)
        assert promotion is not None
        self.assertEqual(promotion.status, PromotionStatus.AWAITING_APPROVAL)
        target = Path(promotion.target_root)
        self.assertFalse((target / "app.py").exists())

        delivered = await service.approve(
            promotion.promotion_id,
            actor="human:test",
        )
        self.assertEqual(delivered.status, PromotionStatus.DELIVERED)
        self.assertEqual(
            (target / "app.py").read_text(encoding="utf-8"),
            "print('reviewed')\n",
        )
        self.assertTrue((target / ".git").is_dir())
        self.assertIsNotNone(delivered.delivered_commit)
        self.assertEqual(
            await service.approve(promotion.promotion_id, actor="human:test"),
            delivered,
        )

    async def test_rejection_never_writes_user_workspace(self) -> None:
        layout = self.accepted_layout("evt-reject")
        service = self.service(DeliveryApprovalMode.HUMAN)
        promotion = await service.prepare_from_run(
            layout=layout,
            conversation_id="conv-reject",
            final_status="COMPLETED",
        )
        assert promotion is not None
        rejected = await service.reject(
            promotion.promotion_id,
            actor="human:test",
            reason="Needs another color.",
        )
        self.assertEqual(rejected.status, PromotionStatus.REJECTED)
        self.assertFalse((Path(rejected.target_root) / "app.py").exists())

    async def test_auto_policy_delivers_after_the_same_review_boundary(self) -> None:
        layout = self.accepted_layout("evt-auto")
        service = self.service(DeliveryApprovalMode.AUTO)
        promotion = await service.prepare_from_run(
            layout=layout,
            conversation_id="conv-auto",
            final_status="COMPLETED",
        )
        assert promotion is not None
        delivered = await service.apply_auto_policy(promotion)
        self.assertEqual(delivered.status, PromotionStatus.DELIVERED)
        self.assertTrue((Path(delivered.target_root) / "app.py").is_file())

    async def test_prepare_is_idempotent_for_one_run(self) -> None:
        layout = self.accepted_layout("evt-idempotent")
        service = self.service(DeliveryApprovalMode.HUMAN)
        first = await service.prepare_from_run(
            layout=layout,
            conversation_id="conv-idempotent",
            final_status="COMPLETED",
        )
        second = await service.prepare_from_run(
            layout=layout,
            conversation_id="conv-idempotent",
            final_status="COMPLETED",
        )
        self.assertEqual(first, second)

    async def test_startup_reconciles_files_copied_before_delivery_commit(self) -> None:
        layout = self.accepted_layout("evt-recovery")
        service = self.service(DeliveryApprovalMode.HUMAN)
        promotion = await service.prepare_from_run(
            layout=layout,
            conversation_id="conv-recovery",
            final_status="COMPLETED",
        )
        assert promotion is not None
        approved = transition_workspace_promotion(
            promotion,
            PromotionStatus.APPROVED,
            decided_by="human:test",
        )
        approved = await self.store.compare_and_set_workspace_promotion(
            expected_status=PromotionStatus.AWAITING_APPROVAL,
            promotion=approved,
        )
        promoting = transition_workspace_promotion(
            approved,
            PromotionStatus.PROMOTING,
        )
        promoting = await self.store.compare_and_set_workspace_promotion(
            expected_status=PromotionStatus.APPROVED,
            promotion=promoting,
        )

        service._copy_files(promoting)
        outcomes = await service.reconcile_promoting()

        self.assertEqual(outcomes[0]["status"], PromotionStatus.DELIVERED.value)
        recovered = await self.store.get_workspace_promotion(
            promotion.promotion_id
        )
        assert recovered is not None
        self.assertEqual(recovered.status, PromotionStatus.DELIVERED)
        self.assertEqual(
            (Path(recovered.target_root) / "app.py").read_text(encoding="utf-8"),
            "print('reviewed')\n",
        )

    async def test_incomplete_final_review_creates_no_request(self) -> None:
        layout = self.accepted_layout("evt-partial")
        service = self.service(DeliveryApprovalMode.HUMAN)
        self.assertIsNone(
            await service.prepare_from_run(
                layout=layout,
                conversation_id="conv-partial",
                final_status="PARTIAL",
            )
        )

    async def test_web_only_deliverable_uses_the_same_human_gate(self) -> None:
        layout = initialize_run_workspace(
            "evt-web-only",
            storage_root=self.root / "runs",
            canonical_root=self.root / "legacy-workspace",
        )
        self.add_web_artifact(layout)
        service = self.service(DeliveryApprovalMode.HUMAN)

        promotion = await service.prepare_from_run(
            layout=layout,
            conversation_id="conv-web-only",
            final_status="COMPLETED",
        )
        assert promotion is not None
        self.assertEqual(promotion.status, PromotionStatus.AWAITING_APPROVAL)
        self.assertIsNone(promotion.source_commit)
        self.assertEqual(promotion.files[0].path, "downloads/template.html")

        delivered = await service.approve(
            promotion.promotion_id,
            actor="human:test",
        )
        target = Path(delivered.target_root) / "downloads" / "template.html"
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "<main>reviewed web artifact</main>",
        )

    async def test_internal_handoff_does_not_become_user_delivery(self) -> None:
        layout = initialize_run_workspace(
            "evt-web-internal",
            storage_root=self.root / "runs",
            canonical_root=self.root / "legacy-workspace",
        )
        self.add_web_artifact(layout, disposition="INTERNAL_HANDOFF")
        service = self.service(DeliveryApprovalMode.HUMAN)

        self.assertIsNone(
            await service.prepare_from_run(
                layout=layout,
                conversation_id="conv-web-internal",
                final_status="COMPLETED",
            )
        )

    async def test_code_and_web_files_share_one_promotion(self) -> None:
        layout = self.accepted_layout("evt-mixed-delivery")
        self.add_web_artifact(layout)
        service = self.service(DeliveryApprovalMode.HUMAN)

        promotion = await service.prepare_from_run(
            layout=layout,
            conversation_id="conv-mixed-delivery",
            final_status="COMPLETED",
        )
        assert promotion is not None
        self.assertEqual(
            [item.path for item in promotion.files],
            ["app.py", "downloads/template.html"],
        )
        delivered = await service.approve(
            promotion.promotion_id,
            actor="human:test",
        )
        target = Path(delivered.target_root)
        self.assertTrue((target / "app.py").is_file())
        self.assertTrue((target / "downloads" / "template.html").is_file())


class RunWorkspaceRetentionTests(unittest.TestCase):
    def test_cleanup_removes_bulky_run_data_but_keeps_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            layout = initialize_run_workspace(
                "evt-old",
                storage_root=root / "runs",
                canonical_root=canonical,
            )
            (layout.private_root / "large.bin").write_bytes(b"x")
            (layout.receipts_root / "proof.json").write_text("{}", encoding="utf-8")
            (canonical / "deliverable.txt").write_text("keep", encoding="utf-8")
            now = datetime.now(timezone.utc)
            outcomes = cleanup_expired_run_workspaces(
                terminal_runs={"evt-old": now - timedelta(days=11)},
                protected_run_ids=set(),
                retention_minutes=10 * 24 * 60,
                storage_root=root / "runs",
                now=now,
            )
            self.assertEqual(outcomes[0]["status"], "CLEANED")
            self.assertFalse(layout.private_root.exists())
            self.assertTrue((layout.receipts_root / "proof.json").is_file())
            self.assertTrue((layout.receipts_root / "retention-cleanup.json").is_file())
            self.assertEqual(
                (canonical / "deliverable.txt").read_text(encoding="utf-8"),
                "keep",
            )


if __name__ == "__main__":
    unittest.main()
