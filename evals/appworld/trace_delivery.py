"""Verify a local Phoenix root span reached SQLite before stopping its owner.

An OTLP HTTP success/SDK force_flush confirms delivery to the collector, not
necessarily completion of the collector's asynchronous database writes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time
import uuid


def wait_for_root(database, trace_id, root_name="appworld.trial", timeout=15):
    started = time.monotonic()
    while True:
        if database.is_file():
            with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as db:
                db.execute("PRAGMA query_only = ON")
                row = db.execute(
                    "SELECT s.span_id FROM spans s JOIN traces t ON t.id=s.trace_rowid "
                    "WHERE t.trace_id=? AND s.name=?", (trace_id, root_name)).fetchone()
                if row:
                    return {"root_persisted": True, "wait_seconds": round(time.monotonic()-started, 3)}
        if time.monotonic() - started >= timeout:
            return {"root_persisted": False, "wait_seconds": round(time.monotonic()-started, 3)}
        time.sleep(0.25)


def main():
    from trace_presentation import run_name
    os.environ.update({
        "PHOENIX_TRACING_ENABLED": "true", "PHOENIX_HOST": "127.0.0.1", "PHOENIX_PORT": "6007",
        "PHOENIX_OTEL_PROTOCOL": "http/protobuf", "PHOENIX_PROJECT": run_name("Delivery Check"),
        "PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:6007/v1/traces",
        "PHOENIX_TELEMETRY_ENABLED": "false", "PHOENIX_DISABLE_AGENT_ASSISTANT": "true",
        "PHOENIX_ALLOWED_SANDBOX_PROVIDERS": "NONE",
        "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
    })
    from phoenix_runtime import PhoenixServerRuntime
    from observability import setup_observability, trace_span
    runtime = PhoenixServerRuntime()
    try:
        runtime.start()
        provider = setup_observability()
        if provider is None:
            raise RuntimeError("Phoenix tracer unavailable")
        root_name = "appworld.trace_delivery_smoke." + uuid.uuid4().hex
        with trace_span(root_name, kind="chain") as span:
            if span is None:
                raise RuntimeError("Phoenix root span unavailable")
            trace_id = format(span.get_span_context().trace_id, "032x")
        flushed = provider.force_flush(timeout_millis=10000)
        result = {"model_calls": 0, "sdk_flush_acknowledged": flushed,
                  **wait_for_root(runtime.database_path, trace_id, root_name)}
        path = Path(__file__).resolve().parents[2] / ".agent/evaluations/phoenix-delivery-smoke.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        if not result["root_persisted"]:
            raise SystemExit(1)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
