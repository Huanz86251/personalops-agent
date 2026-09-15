"""Tests for run-scoped General Worker shared handoff wiring."""

import unittest
from pathlib import Path

from run_workspace import initialize_run_workspace
from workers.general_runtime import GeneralStepRuntime


class _EventStore:
    pass


class _Graph:
    async def astream(self, input_state, **kwargs):
        yield "values", input_state


class _CheckpointState:
    created_at = "2026-09-04T00:00:00Z"
    next = ()
    values = {
        "worker_id": "general-resumed",
        "event_id": "run-general-resumed",
        "final_answer": "resumed result",
    }


class _ResumableGraph:
    def __init__(self):
        self.stream_inputs = []

    async def aget_state(self, config):
        return _CheckpointState()

    async def astream(self, input_state, **kwargs):
        self.stream_inputs.append(input_state)
        if False:
            yield None


class GeneralStepRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_general_runtime_receives_read_only_run_handoff(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = initialize_run_workspace(
                "run-general",
                storage_root=root,
            )
            published = layout.handoff_root / "artifact-one" / "copy.md"
            published.parent.mkdir(parents=True)
            published.write_text("shared copy", encoding="utf-8")
            captured = {}

            def worker_factory(model, **kwargs):
                captured.update(kwargs)
                return _Graph()

            runtime = GeneralStepRuntime(
                "offline-model",
                event_store=_EventStore(),
                tools=[],
                progress_every_tool_calls=4,
                run_storage_root=root,
                worker_factory=worker_factory,
            )
            result = await runtime.ainvoke(
                {"worker_id": "general-1", "event_id": "run-general"}
            )

            self.assertEqual(result["worker_id"], "general-1")
            download = captured["backend"].download_files(
                ["/handoff/artifact-one/copy.md"]
            )[0]
            self.assertEqual(download.content, b"shared copy")
            permission = captured["permissions"][0]
            self.assertEqual(permission.mode, "deny")
            self.assertEqual(permission.operations, ["write"])
            self.assertEqual(permission.paths, ["/handoff/**"])

    async def test_process_resume_uses_existing_worker_checkpoint(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            graph = _ResumableGraph()
            runtime = GeneralStepRuntime(
                "offline-model",
                event_store=_EventStore(),
                tools=[],
                progress_every_tool_calls=4,
                run_storage_root=Path(directory),
                worker_factory=lambda *args, **kwargs: graph,
            )

            result = await runtime.ainvoke(
                {
                    "worker_id": "general-resumed",
                    "event_id": "run-general-resumed",
                },
                config={
                    "configurable": {
                        "thread_id": "general-checkpoint-thread",
                        "resume_from_checkpoint": True,
                    }
                },
            )

            self.assertEqual(graph.stream_inputs, [None])
            self.assertEqual(result["final_answer"], "resumed result")
