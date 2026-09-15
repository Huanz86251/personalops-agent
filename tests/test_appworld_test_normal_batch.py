import json
from pathlib import Path

import pytest

from evals.appworld import run_test_normal_batch as batch_runner
from evals.appworld.run_test_normal_batch import split_task_ids


def test_seeded_halves_are_disjoint_complete_and_repeatable():
    task_ids = [f"task-{index}" for index in range(9)]
    first_a, first_b = split_task_ids(task_ids, 42)
    second_a, second_b = split_task_ids(list(reversed(task_ids)), 42)
    assert (first_a, first_b) == (second_a, second_b)
    assert len(first_a) == 4
    assert len(first_b) == 5
    assert set(first_a).isdisjoint(first_b)
    assert set(first_a) | set(first_b) == set(task_ids)


def test_duplicate_ids_are_rejected():
    try:
        split_task_ids(["same", "same"], 1)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate task IDs must fail closed")


def test_parallelism_is_bounded():
    assert batch_runner._validate_parallelism(1) == 1
    assert batch_runner._validate_parallelism(batch_runner.MAX_PARALLELISM) == batch_runner.MAX_PARALLELISM
    for value in (0, batch_runner.MAX_PARALLELISM + 1, True):
        with pytest.raises(ValueError, match="Parallelism"):
            batch_runner._validate_parallelism(value)


def test_half_runs_tasks_with_bounded_parallel_processes(tmp_path, monkeypatch):
    task_ids = [f"task-{index}" for index in range(1, 7)]
    batch = tmp_path / "parallel-batch"
    batch.mkdir()
    manifest = {
        "schema": "appworld-test-normal-halves",
        "split": "test_normal",
        "seed": 42,
        "image": "frozen-image",
        "source_digest": "frozen-source",
        "halves": {
            "A": {
                "task_ids": task_ids,
                "selection_sha256": batch_runner._selection_digest(task_ids),
            },
            "B": {"task_ids": [], "selection_sha256": batch_runner._selection_digest([])},
        },
    }
    (batch / "manifest.private.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(batch_runner, "_source_digest", lambda: "frozen-source")
    monkeypatch.setattr(batch_runner, "_ensure_phoenix_ready", lambda: object())
    monkeypatch.setattr(batch_runner.time, "sleep", lambda _seconds: None)

    launched_commands = []

    class FakeProcess:
        live = 0
        peak = 0
        next_pid = 1000

        def __init__(self, command, **_kwargs):
            launched_commands.append(command)
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.returncode = None
            self.remaining_polls = 2
            FakeProcess.live += 1
            FakeProcess.peak = max(FakeProcess.peak, FakeProcess.live)

            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            (output / "result.private.json").write_text(
                json.dumps({"official_evaluation": {"success": True}}),
                encoding="utf-8",
            )

        def poll(self):
            if self.remaining_polls:
                self.remaining_polls -= 1
                return None
            if self.returncode is None:
                self.returncode = 0
                FakeProcess.live -= 1
            return self.returncode

    monkeypatch.setattr(batch_runner.subprocess, "Popen", FakeProcess)
    import evals.appworld.batch_monitor as batch_monitor

    indexed = []
    monkeypatch.setattr(
        batch_monitor,
        "record_trial",
        lambda trial, monitor_batch: indexed.append((trial, monitor_batch)),
    )

    batch_runner.run_half(
        batch,
        "A",
        allow_paid=True,
        parallelism=3,
        phoenix_project="Shared Project",
    )

    assert FakeProcess.peak == 3
    assert FakeProcess.live == 0
    assert len(launched_commands) == len(task_ids)
    assert all("--no-monitor" in command for command in launched_commands)
    assert all("--monitor-batch" not in command for command in launched_commands)
    assert len(indexed) == len(task_ids)

    results = json.loads((batch / "half-a-results.private.json").read_text(encoding="utf-8"))
    assert [row["index"] for row in results] == list(range(1, len(task_ids) + 1))
    assert all(row["outcome"] == "PASSED" for row in results)
    progress = json.loads((batch / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "FINISHED"
    assert progress["parallelism"] == 3
    assert progress["completed"] == len(task_ids)
    assert progress["active_task_indices"] == []
