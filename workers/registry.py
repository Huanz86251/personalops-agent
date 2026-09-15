"""Capability-based Worker runtime lookup for the Planning Graph."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from planning_models import WorkerKind


SUPPORTED_WORKER_KINDS = frozenset({"GENERAL", "WEB", "CODE"})


class WorkerRuntimeUnavailableError(RuntimeError):
    """Raised when a planned Worker kind has no configured runtime."""


class WorkerAgentRegistry:
    """Immutable mapping from planning capability names to Worker graphs.

    The Planning Graph depends on this registry rather than on Deep Agents,
    Docker, or a concrete model provider. Multiple kinds may intentionally
    share one runtime during an incremental migration.
    """

    def __init__(self, agents: Mapping[WorkerKind | str, Any]) -> None:
        normalized: dict[str, Any] = {}
        for raw_kind, agent in agents.items():
            kind = str(raw_kind).strip().upper()
            if kind not in SUPPORTED_WORKER_KINDS:
                raise ValueError(f"Unsupported Worker kind: {raw_kind}")
            if agent is None:
                raise ValueError(f"Worker runtime cannot be None: {kind}")
            if kind in normalized:
                raise ValueError(f"Duplicate Worker runtime: {kind}")
            normalized[kind] = agent
        if not normalized:
            raise ValueError("At least one Worker runtime must be registered.")
        self._agents = MappingProxyType(normalized)

    @classmethod
    def general_worker_first(cls, general_worker: Any) -> "WorkerAgentRegistry":
        """Use one Deep Agent for every kind during the V1 migration.

        CODE remains SINGLE-only and has no shell sandbox yet. Registering a
        dedicated CODE runtime later replaces only this mapping boundary.
        """

        return cls(
            {
                "GENERAL": general_worker,
                "WEB": general_worker,
                "CODE": general_worker,
            }
        )

    @classmethod
    def with_code_worker(
        cls,
        general_worker: Any,
        code_worker: Any,
    ) -> "WorkerAgentRegistry":
        """Route CODE Steps to a dedicated runtime while preserving V1 Web."""

        return cls(
            {
                "GENERAL": general_worker,
                "WEB": general_worker,
                "CODE": code_worker,
            }
        )

    @classmethod
    def with_specialized_workers(
        cls,
        general_worker: Any,
        web_worker: Any,
        code_worker: Any,
    ) -> "WorkerAgentRegistry":
        """Route each planning capability to its explicit runtime."""

        return cls(
            {
                "GENERAL": general_worker,
                "WEB": web_worker,
                "CODE": code_worker,
            }
        )

    @property
    def registered_kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._agents))

    def require(self, kind: WorkerKind | str) -> Any:
        normalized = str(kind).strip().upper()
        try:
            return self._agents[normalized]
        except KeyError as error:
            raise WorkerRuntimeUnavailableError(
                f"No runtime is registered for Worker kind {normalized}."
            ) from error


__all__ = [
    "SUPPORTED_WORKER_KINDS",
    "WorkerAgentRegistry",
    "WorkerRuntimeUnavailableError",
]
