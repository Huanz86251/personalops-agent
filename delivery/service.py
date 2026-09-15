"""Promotion service from reviewed run integration to conversation workspace."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from git import Repo

from artifact_publisher import ArtifactPublicationReceipt
from delivery.models import (
    DeliveryApprovalMode,
    PromotionStatus,
    WorkspaceFileManifestEntry,
    WorkspacePromotion,
    create_workspace_promotion,
    transition_workspace_promotion,
)
from integration_repository import IntegrationCommitReceipt, IntegrationRepository
from path import AGENT_DATA_ROOT
from run_workspace import (
    CONVERSATION_WORKSPACE_ROOT,
    RunWorkspaceLayout,
    initialize_conversation_workspace,
    stable_storage_key,
)
from workers.code_publisher import compute_code_tree_revision

if TYPE_CHECKING:
    from eventing.store import AsyncEventStore


class WorkspacePromotionService:
    """Keep model judgment separate from deterministic user-file delivery."""

    def __init__(
        self,
        *,
        event_store: "AsyncEventStore",
        approval_mode: DeliveryApprovalMode,
        conversation_workspace_root: Path = CONVERSATION_WORKSPACE_ROOT,
        metadata_root: Path = AGENT_DATA_ROOT / "conversation_workspaces",
    ) -> None:
        self.event_store = event_store
        self.approval_mode = approval_mode
        self.conversation_workspace_root = Path(conversation_workspace_root).resolve()
        self.metadata_root = Path(metadata_root).resolve()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _accepted_commits_from_receipts_root(
        receipts_root: Path,
    ) -> tuple[IntegrationCommitReceipt, ...]:
        root = Path(receipts_root).resolve() / "integration-commits"
        if not root.is_dir():
            return ()
        receipts = [
            IntegrationCommitReceipt.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            for path in sorted(root.glob("accepted-*.json"))
        ]
        return tuple(sorted(receipts, key=lambda item: item.committed_at))

    @classmethod
    def _accepted_commits(
        cls,
        layout: RunWorkspaceLayout,
    ) -> tuple[IntegrationCommitReceipt, ...]:
        return cls._accepted_commits_from_receipts_root(layout.receipts_root)

    def _code_manifest(
        self,
        *,
        layout: RunWorkspaceLayout,
        receipts: tuple[IntegrationCommitReceipt, ...],
    ) -> tuple[WorkspaceFileManifestEntry, ...]:
        approved_paths = sorted(
            {
                path
                for receipt in receipts
                for path in receipt.changed_files
            }
        )
        entries: list[WorkspaceFileManifestEntry] = []
        for relative in approved_paths:
            source = (layout.integration_root / relative).resolve()
            if not source.is_relative_to(layout.integration_root.resolve()):
                raise ValueError("accepted path escapes integration root")
            if not source.is_file():
                raise ValueError(
                    f"accepted integration file is missing: {relative}"
                )
            entries.append(
                WorkspaceFileManifestEntry(
                    source_path=f"integration/{relative}",
                    path=relative,
                    size_bytes=source.stat().st_size,
                    sha256=self._sha256(source),
                )
            )
        return tuple(entries)

    @staticmethod
    def _user_deliverable_receipts(
        layout: RunWorkspaceLayout,
    ) -> tuple[ArtifactPublicationReceipt, ...]:
        receipts: list[ArtifactPublicationReceipt] = []
        for path in sorted(layout.receipts_root.glob("handoff-*.json")):
            receipt = ArtifactPublicationReceipt.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if receipt.run_id != layout.run_id:
                raise ValueError("handoff receipt belongs to another run")
            if receipt.disposition == "USER_DELIVERABLE":
                receipts.append(receipt)
        return tuple(sorted(receipts, key=lambda item: item.published_at))

    def _web_manifest(
        self,
        *,
        layout: RunWorkspaceLayout,
        receipts: tuple[ArtifactPublicationReceipt, ...],
    ) -> tuple[WorkspaceFileManifestEntry, ...]:
        entries: list[WorkspaceFileManifestEntry] = []
        run_root = layout.run_root.resolve()
        handoff_root = layout.handoff_root.resolve()
        for receipt in receipts:
            target_path = str(receipt.target_path or "").strip()
            if not target_path:
                raise ValueError(
                    f"user deliverable has no target path: {receipt.publication_id}"
                )
            source = Path(receipt.storage_path).resolve()
            if not source.is_relative_to(handoff_root) or not source.is_file():
                raise ValueError(
                    f"user deliverable escaped run handoff: {receipt.publication_id}"
                )
            if source.stat().st_size != receipt.size_bytes:
                raise ValueError(
                    f"user deliverable size changed: {receipt.publication_id}"
                )
            digest = self._sha256(source)
            if digest != receipt.sha256:
                raise ValueError(
                    f"user deliverable hash changed: {receipt.publication_id}"
                )
            entries.append(
                WorkspaceFileManifestEntry(
                    source_path=source.relative_to(run_root).as_posix(),
                    path=target_path,
                    size_bytes=receipt.size_bytes,
                    sha256=receipt.sha256,
                )
            )
        return tuple(entries)

    async def prepare_from_run(
        self,
        *,
        layout: RunWorkspaceLayout,
        conversation_id: str,
        final_status: str,
    ) -> WorkspacePromotion | None:
        """Create a durable request only for a fully accepted CODE run."""

        if str(final_status).strip().upper() != "COMPLETED":
            return None
        existing_for_run = await self.event_store.list_workspace_promotions(
            run_id=layout.run_id,
            limit=2,
        )
        if existing_for_run:
            if len(existing_for_run) != 1:
                raise RuntimeError("one run owns multiple promotion requests")
            return existing_for_run[0]
        code_receipts = self._accepted_commits(layout)
        web_receipts = self._user_deliverable_receipts(layout)
        if not code_receipts and not web_receipts:
            return None
        source_commit: str | None = None
        source_commit_root: str | None = None
        if code_receipts:
            repository = IntegrationRepository(
                run_id=layout.run_id,
                root=layout.integration_root,
                receipts_root=layout.receipts_root,
            )
            status = repository.status()
            if not status.working_tree_clean:
                raise RuntimeError(
                    "reviewed integration is dirty; delivery request was not created"
                )
            source_commit = status.head_commit
            source_commit_root = "integration"
        files = (
            *self._code_manifest(layout=layout, receipts=code_receipts),
            *self._web_manifest(layout=layout, receipts=web_receipts),
        )
        target_paths = [item.path for item in files]
        if len(target_paths) != len(set(target_paths)):
            raise RuntimeError(
                "reviewed Code and Web artifacts target the same workspace path"
            )
        if not files:
            return None
        target_root = initialize_conversation_workspace(
            conversation_id,
            root=self.conversation_workspace_root,
        )
        target_revision = compute_code_tree_revision(target_root)
        review_summary = " | ".join(
            receipt.review_summary
            for receipt in code_receipts
            if receipt.review_summary
        )[-3000:]
        if web_receipts:
            web_summary = "Step Reporter approved Web deliverables: " + ", ".join(
                receipt.description or receipt.output_id or receipt.candidate_id
                for receipt in web_receipts
            )
            review_summary = " | ".join(
                item for item in (review_summary, web_summary) if item
            )
        review_summary = review_summary[-4000:] or "Independent review accepted the files."
        test_summary = " | ".join(
            receipt.test_summary
            for receipt in code_receipts
            if receipt.test_summary
        )[-3000:]
        if web_receipts:
            web_test_summary = (
                f"Harness verified {len(web_receipts)} Web artifact receipt(s), "
                "file size(s), and SHA-256 hash(es)."
            )
            test_summary = " | ".join(
                item for item in (test_summary, web_test_summary) if item
            )
        test_summary = test_summary[-4000:] or "Reviewer checks passed."
        promotion = create_workspace_promotion(
            event_id=layout.run_id,
            conversation_id=conversation_id,
            approval_mode=self.approval_mode,
            source_root=str(layout.run_root.resolve()),
            source_commit=source_commit,
            source_commit_root=source_commit_root,
            target_root=str(target_root),
            target_base_revision=target_revision,
            files=files,
            review_summary=review_summary,
            test_summary=test_summary,
        )
        existing = await self.event_store.get_workspace_promotion(
            promotion.promotion_id
        )
        if existing is not None:
            return existing
        pending = await self.event_store.list_workspace_promotions(
            conversation_id=conversation_id,
            statuses=(
                PromotionStatus.AWAITING_APPROVAL,
                PromotionStatus.APPROVED,
            ),
        )
        for previous in pending:
            superseded = transition_workspace_promotion(
                previous,
                PromotionStatus.SUPERSEDED,
            )
            await self.event_store.compare_and_set_workspace_promotion(
                expected_status=previous.status,
                promotion=superseded,
            )
        return await self.event_store.add_workspace_promotion(promotion)

    def _conversation_repository(
        self,
        promotion: WorkspacePromotion,
    ) -> IntegrationRepository:
        receipts_root = (
            self.metadata_root
            / stable_storage_key(promotion.conversation_id, prefix="conversation")
            / "receipts"
        )
        return IntegrationRepository(
            run_id=f"conversation:{promotion.conversation_id}",
            root=Path(promotion.target_root),
            receipts_root=receipts_root,
        )

    def _source_is_valid(self, promotion: WorkspacePromotion) -> None:
        source_root = Path(promotion.source_root).resolve()
        if promotion.source_commit is not None:
            repository_root = (
                source_root / str(promotion.source_commit_root)
            ).resolve()
            if not repository_root.is_relative_to(source_root):
                raise RuntimeError("reviewed repository escaped delivery source")
            with Repo(repository_root) as repository:
                actual_commit = repository.head.commit.hexsha
            if actual_commit != promotion.source_commit:
                raise RuntimeError(
                    "reviewed integration commit changed before delivery"
                )
        for entry in promotion.files:
            source = (
                source_root / (entry.source_path or entry.path)
            ).resolve()
            if not source.is_relative_to(source_root) or not source.is_file():
                raise RuntimeError(f"delivery source is missing: {entry.path}")
            if source.stat().st_size != entry.size_bytes or self._sha256(source) != entry.sha256:
                raise RuntimeError(f"delivery source hash changed: {entry.path}")

    @staticmethod
    def _copy_files(
        promotion: WorkspacePromotion,
    ) -> None:
        source_root = Path(promotion.source_root).resolve()
        target_root = Path(promotion.target_root).resolve()
        with tempfile.TemporaryDirectory(
            prefix=".personalops-delivery-",
            dir=target_root.parent,
        ) as temporary:
            staged = Path(temporary)
            for entry in promotion.files:
                source = (
                    source_root / (entry.source_path or entry.path)
                ).resolve()
                destination = staged / entry.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            backups = staged / ".backups"
            applied: list[Path] = []
            try:
                for entry in promotion.files:
                    destination = (target_root / entry.path).resolve()
                    if not destination.is_relative_to(target_root):
                        raise RuntimeError("delivery target escaped conversation workspace")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        if not destination.is_file():
                            raise RuntimeError(
                                f"delivery target is not a file: {entry.path}"
                            )
                        backup = backups / entry.path
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(destination, backup)
                    os.replace(staged / entry.path, destination)
                    applied.append(destination)
            except BaseException:
                for destination in reversed(applied):
                    relative = destination.relative_to(target_root)
                    backup = backups / relative
                    if backup.is_file():
                        os.replace(backup, destination)
                    else:
                        destination.unlink(missing_ok=True)
                raise

    def _files_match_target(self, promotion: WorkspacePromotion) -> bool:
        target_root = Path(promotion.target_root).resolve()
        return all(
            (target_root / entry.path).is_file()
            and (target_root / entry.path).stat().st_size == entry.size_bytes
            and self._sha256(target_root / entry.path) == entry.sha256
            for entry in promotion.files
        )

    async def approve(
        self,
        promotion_id: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> WorkspacePromotion:
        promotion = await self.event_store.get_workspace_promotion(promotion_id)
        if promotion is None:
            raise LookupError(f"Unknown promotion_id: {promotion_id}")
        if promotion.status is PromotionStatus.DELIVERED:
            return promotion
        if promotion.status is not PromotionStatus.AWAITING_APPROVAL:
            raise RuntimeError(
                f"promotion cannot be approved from {promotion.status.value}"
            )
        approved = transition_workspace_promotion(
            promotion,
            PromotionStatus.APPROVED,
            decided_by=actor,
            decision_reason=reason,
        )
        await self.event_store.compare_and_set_workspace_promotion(
            expected_status=promotion.status,
            promotion=approved,
        )
        return await self.deliver(approved.promotion_id)

    async def reject(
        self,
        promotion_id: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> WorkspacePromotion:
        promotion = await self.event_store.get_workspace_promotion(promotion_id)
        if promotion is None:
            raise LookupError(f"Unknown promotion_id: {promotion_id}")
        if promotion.status is PromotionStatus.REJECTED:
            return promotion
        if promotion.status is not PromotionStatus.AWAITING_APPROVAL:
            raise RuntimeError(
                f"promotion cannot be rejected from {promotion.status.value}"
            )
        rejected = transition_workspace_promotion(
            promotion,
            PromotionStatus.REJECTED,
            decided_by=actor,
            decision_reason=reason or "User rejected delivery.",
        )
        return await self.event_store.compare_and_set_workspace_promotion(
            expected_status=promotion.status,
            promotion=rejected,
        )

    async def deliver(self, promotion_id: str) -> WorkspacePromotion:
        promotion = await self.event_store.get_workspace_promotion(promotion_id)
        if promotion is None:
            raise LookupError(f"Unknown promotion_id: {promotion_id}")
        if promotion.status is PromotionStatus.DELIVERED:
            return promotion
        if promotion.status is PromotionStatus.APPROVED:
            promoting = transition_workspace_promotion(
                promotion,
                PromotionStatus.PROMOTING,
            )
            promotion = await self.event_store.compare_and_set_workspace_promotion(
                expected_status=PromotionStatus.APPROVED,
                promotion=promoting,
            )
        if promotion.status is not PromotionStatus.PROMOTING:
            raise RuntimeError(
                f"promotion cannot be delivered from {promotion.status.value}"
            )
        try:
            self._source_is_valid(promotion)
            repository = self._conversation_repository(promotion)
            repository.initialize()
            actual_base = compute_code_tree_revision(Path(promotion.target_root))
            if actual_base == promotion.target_base_revision:
                self._copy_files(promotion)
            elif not self._files_match_target(promotion):
                raise RuntimeError(
                    "conversation workspace changed after approval request"
                )
            source_receipts = self._accepted_commits_from_receipts_root(
                Path(promotion.source_root) / "receipts"
            )
            latest_source_receipt = source_receipts[-1] if source_receipts else None
            receipt = repository.commit_accepted(
                step_id=(latest_source_receipt.step_id if latest_source_receipt else 1),
                candidate_revision=(
                    latest_source_receipt.candidate_revision
                    if latest_source_receipt
                    else 1
                ),
                manifest_id=promotion.manifest_id,
                publication_id=promotion.promotion_id,
                approved_paths=tuple(entry.path for entry in promotion.files),
                review_summary=promotion.review_summary,
                test_summary=promotion.test_summary,
            )
            delivered = transition_workspace_promotion(
                promotion,
                PromotionStatus.DELIVERED,
                delivered_commit=receipt.accepted_commit,
            )
            return await self.event_store.compare_and_set_workspace_promotion(
                expected_status=PromotionStatus.PROMOTING,
                promotion=delivered,
            )
        except Exception as error:
            failed = transition_workspace_promotion(
                promotion,
                PromotionStatus.FAILED,
                failure_reason=f"{type(error).__name__}: {error}",
            )
            await self.event_store.compare_and_set_workspace_promotion(
                expected_status=PromotionStatus.PROMOTING,
                promotion=failed,
            )
            raise

    async def apply_auto_policy(
        self,
        promotion: WorkspacePromotion,
    ) -> WorkspacePromotion:
        if promotion.approval_mode is not DeliveryApprovalMode.AUTO:
            return promotion
        if promotion.status is PromotionStatus.AWAITING_APPROVAL:
            return await self.approve(
                promotion.promotion_id,
                actor="policy:auto",
                reason="Configured automatic delivery after Code Reviewer approval.",
            )
        if promotion.status in {
            PromotionStatus.APPROVED,
            PromotionStatus.PROMOTING,
        }:
            return await self.deliver(promotion.promotion_id)
        return promotion

    async def reconcile_promoting(self) -> tuple[dict[str, str], ...]:
        """Recover each interrupted delivery independently on process startup."""

        outcomes: list[dict[str, str]] = []
        promotions = await self.event_store.list_workspace_promotions(
            statuses=(PromotionStatus.PROMOTING,),
        )
        for promotion in promotions:
            try:
                delivered = await self.deliver(promotion.promotion_id)
                outcomes.append(
                    {
                        "promotion_id": promotion.promotion_id,
                        "status": delivered.status.value,
                    }
                )
            except Exception as error:
                outcomes.append(
                    {
                        "promotion_id": promotion.promotion_id,
                        "status": PromotionStatus.FAILED.value,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
        return tuple(outcomes)


def format_promotion_summary(promotion: WorkspacePromotion) -> str:
    lines = [
        "文件已经通过独立 Reviewer 验收，正在等待交付确认。",
        "",
        f"交付编号：{promotion.promotion_id}",
        f"目标目录：{promotion.target_root}",
        "",
        "文件变化：",
        *[f"- {item.path} ({item.size_bytes} bytes)" for item in promotion.files],
        "",
        f"验证摘要：{promotion.test_summary}",
        f"Reviewer：{promotion.review_summary}",
        "",
        f"批准：/approve {promotion.promotion_id}",
        f"拒绝：/reject {promotion.promotion_id} 原因",
    ]
    return "\n".join(lines)


__all__ = [
    "WorkspacePromotionService",
    "format_promotion_summary",
]
