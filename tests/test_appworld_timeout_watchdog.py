import json
from datetime import datetime, timedelta, timezone

from scripts import watch_appworld_task_timeouts as watchdog


def test_overdue_child_is_verified_stopped_and_logged(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    batch.mkdir()
    now = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    started = now - timedelta(minutes=41)
    trial = batch / "trials" / "A" / "001-task-1"
    trial.mkdir(parents=True)
    (batch / "progress.json").write_text(json.dumps({
        "status": "RUNNING", "half": "A", "active_tasks": [
            {"index": 1, "task_id": "task-1", "started_at_utc": started.isoformat()}
        ],
    }), encoding="utf-8")
    (batch / "half-a.log").write_text("START 1/84 pid=123 active=1/4\n", encoding="utf-8")

    class FakeProcess:
        terminated = False

        def __init__(self, pid):
            assert pid == 123

        def cmdline(self):
            return ["python", "-m", "evals.appworld.run_conversation", "--task", "task-1",
                    "--split", "test_normal", "--output", str(trial)]

        def create_time(self):
            return started.timestamp()

        def is_running(self):
            return True

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(watchdog.psutil, "Process", FakeProcess)
    events = watchdog.stop_overdue(batch, now, 40 * 60)
    assert len(events) == 1
    assert events[0]["classification"] == "TIMEOUT_CENSORED"
    recorded = [json.loads(line) for line in (batch / watchdog.SIDE_CAR).read_text().splitlines()]
    assert [event["event"] for event in recorded] == [
        "wall_timeout_stop_requested", "wall_timeout_terminate_sent",
    ]


def test_wrong_trial_command_is_not_stopped(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    batch.mkdir()
    now = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    started = now - timedelta(minutes=41)
    (batch / "progress.json").write_text(json.dumps({
        "status": "RUNNING", "half": "A", "active_tasks": [
            {"index": 1, "task_id": "task-1", "started_at_utc": started.isoformat()}
        ],
    }), encoding="utf-8")
    (batch / "half-a.log").write_text("START 1/84 pid=123 active=1/4\n", encoding="utf-8")

    class WrongProcess:
        def __init__(self, _pid):
            pass

        def cmdline(self):
            return ["python", "-m", "evals.appworld.run_conversation", "--task", "other",
                    "--split", "test_normal", "--output", str(batch / "other")]

        def create_time(self):
            return started.timestamp()

        def is_running(self):
            return True

        def terminate(self):
            raise AssertionError("Must not touch an unrelated process")

    monkeypatch.setattr(watchdog.psutil, "Process", WrongProcess)
    assert watchdog.stop_overdue(batch, now, 40 * 60) == []
    assert not (batch / watchdog.SIDE_CAR).exists()


def test_challenge_watchdog_does_not_match_normal_child(tmp_path, monkeypatch):
    batch = tmp_path / "challenge"
    batch.mkdir()
    now = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    started = now - timedelta(minutes=41)
    trial = batch / "trials" / "A" / "001-task-1"
    (batch / "progress.json").write_text(json.dumps({
        "status": "RUNNING", "half": "A", "active_tasks": [
            {"index": 1, "task_id": "task-1", "started_at_utc": started.isoformat()}
        ],
    }), encoding="utf-8")
    (batch / "half-a.log").write_text("START 1/2 pid=123 active=1/4\n", encoding="utf-8")

    class NormalProcess:
        def __init__(self, _pid):
            pass

        def cmdline(self):
            return ["python", "-m", "evals.appworld.run_conversation", "--task", "task-1",
                    "--split", "test_normal", "--output", str(trial)]

        def create_time(self):
            return started.timestamp()

        def is_running(self):
            return True

        def terminate(self):
            raise AssertionError("Must not touch a Normal run")

    monkeypatch.setattr(watchdog.psutil, "Process", NormalProcess)
    assert watchdog.stop_overdue(batch, now, 40 * 60, split="test_challenge") == []
