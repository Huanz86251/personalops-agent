"""Deterministic, token-free rules for coalescing Worker progress events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from workers.leadership_models import LeadershipWakeReason
from workers.progress import WorkerProgressRecord


class LeadershipWakePolicy:
    """Apply fast-path triggers before threshold or watermark barriers."""

    def __init__(
        self,
        *,
        single_worker_reports: int = 2,
        multi_worker_reports: int = 1,
    ) -> None:
        if single_worker_reports < 1:
            raise ValueError("single_worker_reports must be positive.")
        if multi_worker_reports < 1:
            raise ValueError("multi_worker_reports must be positive.")
        self.single_worker_reports = single_worker_reports
        self.multi_worker_reports = multi_worker_reports

    def evaluate(
        self,
        *,
        active_worker_ids: Sequence[str],
        reports_by_worker: Mapping[str, Sequence[WorkerProgressRecord]],
    ) -> LeadershipWakeReason | None:
        """Return a wake reason, or None when the harness should auto-continue."""

        active = tuple(dict.fromkeys(active_worker_ids))
        if not active:
            return None

        reports = [
            report
            for worker_id in active
            for report in reports_by_worker.get(worker_id, ())
        ]
        if any(
            report.progress.completion_claim == "blocked"
            for report in reports
        ):
            return "WORKER_BLOCKED"
        if any(
            report.progress.completion_claim == "possibly_ready"
            for report in reports
        ):
            return "WORKER_POSSIBLY_READY"

        if len(active) == 1:
            worker_reports = reports_by_worker.get(active[0], ())
            if len(worker_reports) >= self.single_worker_reports:
                return "SINGLE_WORKER_THRESHOLD"
            return None

        if all(
            len(reports_by_worker.get(worker_id, ()))
            >= self.multi_worker_reports
            for worker_id in active
        ):
            return "MULTI_WORKER_BARRIER"
        return None


__all__ = ["LeadershipWakePolicy"]
