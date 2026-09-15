"""Offline tests for capability-based Worker runtime lookup."""

import unittest

from workers.registry import (
    WorkerAgentRegistry,
    WorkerRuntimeUnavailableError,
)


class WorkerAgentRegistryTests(unittest.TestCase):
    def test_general_worker_can_back_all_kinds_during_migration(self):
        worker = object()
        registry = WorkerAgentRegistry.general_worker_first(worker)

        self.assertIs(registry.require("GENERAL"), worker)
        self.assertIs(registry.require("WEB"), worker)
        self.assertIs(registry.require("CODE"), worker)
        self.assertEqual(
            registry.registered_kinds,
            ("CODE", "GENERAL", "WEB"),
        )

    def test_code_worker_can_be_registered_without_changing_callers(self):
        general_worker = object()
        code_worker = object()
        registry = WorkerAgentRegistry(
            {
                "GENERAL": general_worker,
                "WEB": general_worker,
                "CODE": code_worker,
            }
        )

        self.assertIs(registry.require("CODE"), code_worker)

    def test_specialized_factory_keeps_general_and_web_shared(self):
        general_worker = object()
        code_worker = object()
        registry = WorkerAgentRegistry.with_code_worker(
            general_worker,
            code_worker,
        )

        self.assertIs(registry.require("GENERAL"), general_worker)
        self.assertIs(registry.require("WEB"), general_worker)
        self.assertIs(registry.require("CODE"), code_worker)

    def test_fully_specialized_factory_routes_all_kinds(self):
        general_worker = object()
        web_worker = object()
        code_worker = object()
        registry = WorkerAgentRegistry.with_specialized_workers(
            general_worker,
            web_worker,
            code_worker,
        )

        self.assertIs(registry.require("GENERAL"), general_worker)
        self.assertIs(registry.require("WEB"), web_worker)
        self.assertIs(registry.require("CODE"), code_worker)

    def test_missing_runtime_fails_explicitly(self):
        registry = WorkerAgentRegistry({"GENERAL": object()})

        with self.assertRaises(WorkerRuntimeUnavailableError):
            registry.require("WEB")


if __name__ == "__main__":
    unittest.main()
