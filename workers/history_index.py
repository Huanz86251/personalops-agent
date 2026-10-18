"""Persistent short aliases for immutable, task-scoped tool-call archives."""
import json
import re
import sqlite3
from pathlib import Path


def step_number(record):
    explicit = record.get("step_id")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        return explicit
    match = re.search(r"(?:^|:)step[_:](\d+)(?=:|$)", str(record.get("worker", "")))
    return int(match[1]) if match and int(match[1]) > 0 else None


def indexed_calls(directory, records):
    """Allocate once under a transaction; worker identity disambiguates call IDs."""
    path = Path(directory) / "calls.sqlite3"
    if path.is_symlink():
        raise ValueError("History index cannot be a symbolic link")
    requests = []
    results = {}
    for record in records:
        worker = record["worker"]
        message = record["message"]
        role = message.get("type", message.get("role"))
        if role == "tool":
            key = (worker, message.get("tool_call_id"))
            results.setdefault(key, []).append(message)
        elif role in {"ai", "assistant"}:
            step = step_number(record)
            if step is None:
                continue
            for call in message.get("tool_calls", []):
                if call.get("id") and call.get("name"):
                    requests.append((step, worker, call, record["reference"]))

    connection = sqlite3.connect(path, timeout=30)
    try:
        connection.execute("""CREATE TABLE IF NOT EXISTS calls (
            step_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
            worker TEXT NOT NULL, call_id TEXT NOT NULL,
            call_json TEXT NOT NULL, source_reference TEXT NOT NULL,
            PRIMARY KEY(worker, call_id), UNIQUE(step_id, ordinal))""")
        connection.execute("BEGIN IMMEDIATE")
        for step, worker, call, source in requests:
            exists = connection.execute(
                "SELECT 1 FROM calls WHERE worker=? AND call_id=?", (worker, call["id"])
            ).fetchone()
            if exists:
                continue
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(ordinal),0)+1 FROM calls WHERE step_id=?", (step,)
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO calls VALUES(?,?,?,?,?,?)",
                (step, ordinal, worker, call["id"], json.dumps(call, ensure_ascii=False), source),
            )
        connection.commit()
        rows = connection.execute(
            "SELECT step_id,ordinal,worker,call_id,call_json FROM calls ORDER BY step_id,ordinal"
        ).fetchall()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    available = {(worker, call["id"]) for _, worker, call, _ in requests}
    calls = []
    for step, ordinal, worker, call_id, raw in rows:
        key = (worker, call_id)
        if key not in available:
            continue
        call = json.loads(raw)
        calls.append({
            "reference": f"S{step}-{ordinal}", "step_id": step, "worker": worker,
            "tool_name": call["name"], "arguments": call.get("args", {}),
            "tool_results": results.get(key, []),
        })
    return calls
