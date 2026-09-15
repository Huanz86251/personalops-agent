"""Zero-model acceptance: real Docker/AppWorld grader plus durable Phoenix trace.

This checks infrastructure, not Agent quality. The arithmetic probe deliberately
leaves the Train task unsolved. No provider credentials are loaded or used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


def main():
    from trace_presentation import run_name
    os.environ.update({
        "PHOENIX_TRACING_ENABLED": "true", "PHOENIX_HOST": "127.0.0.1", "PHOENIX_PORT": "6007",
        "PHOENIX_OTEL_PROTOCOL": "http/protobuf", "PHOENIX_PROJECT": run_name("Infrastructure Check"),
        "PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:6007/v1/traces",
        "PHOENIX_TELEMETRY_ENABLED": "false", "PHOENIX_DISABLE_AGENT_ASSISTANT": "true",
        "PHOENIX_ALLOWED_SANDBOX_PROVIDERS": "NONE",
        "PHOENIX_TRACE_PROFILE": "curated",
        "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
    })
    from phoenix_runtime import PhoenixServerRuntime
    from observability import (setup_observability, trace_context, trace_span,
                               set_span_attributes, set_span_output)
    from .smoke import check_world
    from .trace_delivery import wait_for_root

    runtime = PhoenixServerRuntime()
    check_id = "acceptance_" + uuid.uuid4().hex
    root_name = "EVAL / Infrastructure acceptance"
    directory = Path(__file__).resolve().parents[2] / ".agent/evaluations" / check_id
    directory.mkdir(parents=True, exist_ok=False)
    report = {"check_id": check_id, "kind": "zero_model_infrastructure_acceptance",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "external_model_calls": 0, "passed": False}
    try:
        runtime.start()
        provider = setup_observability()
        if provider is None:
            raise RuntimeError("Phoenix tracer unavailable")
        with trace_context(session_id=check_id, metadata={"purpose": "infrastructure_acceptance"}):
            with trace_span(root_name, kind="agent", input_value={
                    "purpose": "zero-model compatibility check",
                    "external_model_calls": 0,
                }) as span:
                if span is None:
                    raise RuntimeError("Phoenix span unavailable")
                report["trace_id"] = format(span.get_span_context().trace_id, "032x")
                report["world"] = check_world()
                set_span_attributes(span, **{
                    "eval.infrastructure_verified": True,
                    "eval.external_model_calls": 0,
                    "eval.official_task_success": report["world"]["official_task_success"],
                })
                set_span_output(span, {
                    "infrastructure_passed": True,
                    "external_model_calls": 0,
                    "persistent_execution_verified": True,
                    "official_evaluator_returned": True,
                    "task_deliberately_unsolved": True,
                })
        report["sdk_flush_acknowledged"] = provider.force_flush(timeout_millis=10000)
        report["before_stop"] = wait_for_root(runtime.database_path, report["trace_id"], root_name)
        runtime.stop()
        report["after_stop"] = wait_for_root(runtime.database_path, report["trace_id"], root_name, timeout=0)
        if not (report["sdk_flush_acknowledged"] and report["before_stop"]["root_persisted"]
                and report["after_stop"]["root_persisted"]):
            raise RuntimeError("Trace was not confirmed durable")
        report["passed"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        runtime.stop()
        (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"passed": report["passed"], "external_model_calls": 0,
                          "result_path": str(directory / "result.json")}, indent=2))


if __name__ == "__main__":
    main()
