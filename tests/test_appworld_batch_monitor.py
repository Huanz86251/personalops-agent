from __future__ import annotations

import base64
import csv
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from evals.appworld.batch_monitor import record_trial, scan_runs, summarize_trial


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_trial(root: Path, name: str, *, success: bool | None, cost: float, task_id: str) -> Path:
    trial = root / name
    trial.mkdir()
    write_json(trial / "manifest.json", {"task_id": task_id, "split": "train"})
    evaluation = None if success is None else {"success": success}
    write_json(trial / "result.private.json", {
        "task_id": task_id,
        "status": "finished",
        "official_evaluation": evaluation,
        "scheduler_status": "COMPLETED" if success else "FAILED",
        "stop_reason": "fixture stop",
        "evaluation_skipped": "no terminal review" if success is None else None,
    })
    write_json(trial / "usage.private.json", {
        "r1": {"model_role": "scheduler", "input_tokens": 100, "output_tokens": 20,
               "cache_read_input_tokens": 50, "reasoning_output_tokens": 5},
        "r2": {"model_role": "general", "input_tokens": 300, "output_tokens": 40,
               "cache_read_input_tokens": 150, "reasoning_output_tokens": 10},
    })
    write_json(trial / "cost-summary.json", {
        "total": {"known_total_cny": cost, "missing_requests": 0}
    })
    write_json(trial / "spans.private.json", [
        {"name": "AppWorld Task", "trace_id": "a" * 32, "span_id": "1" * 16,
         "status": "OK", "start_time": 1_000_000_000, "end_time": 11_000_000_000},
        {"name": "General Agent / Step", "trace_id": "a" * 32, "span_id": "2" * 16,
         "status": "ERROR" if success is False else "OK",
         "start_time": 2_000_000_000, "end_time": 10_000_000_000},
    ])
    return trial


def make_phoenix(path: Path):
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE projects(id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("CREATE TABLE traces(id INTEGER PRIMARY KEY, project_rowid INTEGER, trace_id TEXT)")
        db.execute("INSERT INTO projects(id,name) VALUES(51,'fixture')")
        db.execute("INSERT INTO traces(id,project_rowid,trace_id) VALUES(1,51,?)", ("a" * 32,))
        db.commit()


def test_summarize_trial_uses_official_grade_usage_roles_and_phoenix_link():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        trial = make_trial(root, "trial-failed", success=False, cost=0.12, task_id="train-7")
        database = root / "phoenix.db"
        make_phoenix(database)
        row = summarize_trial(trial, phoenix_db=database)
        assert row["outcome"] == "FAILED"
        assert row["input_tokens"] == 400
        assert row["output_tokens"] == 60
        assert row["cache_read_tokens"] == 200
        assert row["cache_hit_ratio"] == 0.5
        assert row["top_token_role"] == "general"
        assert row["elapsed_seconds"] == 10.0
        assert row["longest_component"] == "General Agent / Step"
        assert row["failure_kind"] == "official_failed,error_span"
        assert row["error_span_count"] == 1
        project_ref = base64.b64encode(b"Project:51").decode()
        assert f"/projects/{project_ref}/spans/" in row["phoenix_url"]


def test_record_trial_builds_incremental_private_json_csv_and_ranked_report():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        runs = root / "runs"
        runs.mkdir()
        output = root / "monitor"
        database = root / "phoenix.db"
        make_phoenix(database)
        passed = make_trial(runs, "trial-pass", success=True, cost=0.03, task_id="train-1")
        failed = make_trial(runs, "trial-fail", success=False, cost=0.20, task_id="train-2")
        record_trial(passed, "batch-a", output_root=output, phoenix_db=database)
        batch = record_trial(failed, "batch-a", output_root=output, phoenix_db=database)

        rows = json.loads((batch / "index.json").read_text(encoding="utf-8"))
        assert {row["outcome"] for row in rows} == {"PASSED", "FAILED"}
        with (batch / "index.csv").open(encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        assert {row["task_id"] for row in csv_rows} == {"train-1", "train-2"}
        report = (batch / "report.md").read_text(encoding="utf-8")
        assert "失败与未评分" in report
        assert "费用最高" in report
        assert "Token最高" in report
        assert "耗时最长" in report
        assert "按角色Token热点" in report
        assert "| 任务 | 运行 | 结果 | 原因 |" in report
        assert "train-2" in report
        assert "fixture stop" not in report
        assert scan_runs(runs) == [failed, passed]
