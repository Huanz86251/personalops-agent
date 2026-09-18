"""Single-task CLI using normal role configuration and canonical Phoenix callbacks.

Run from the repository root: python -m evals.appworld.run_conversation --help
No paid execution occurs without the explicit --allow-paid option.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

EXECUTION_ENVIRONMENT = "appworld"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test_normal", "test_challenge"), required=True)
    parser.add_argument("--image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phoenix-project")
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--max-interactions", type=int, default=90)
    parser.add_argument("--allow-paid", action="store_true")
    monitor_default = os.getenv("APPWORLD_RUN_MONITOR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    parser.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=monitor_default,
                        help="Index this run after completion; use --no-monitor to detach the hook.")
    parser.add_argument("--monitor-batch", default=os.getenv("APPWORLD_MONITOR_BATCH_ID"),
                        help="Group name for the private batch report.")
    args = parser.parse_args()
    if not args.allow_paid:
        parser.error("Use scripts/probe_appworld_conversation.py for free checks; paid execution requires --allow-paid.")
    if not 1 <= args.max_calls <= 100 or not 1 <= args.max_interactions <= 100:
        parser.error("Call and interaction limits must be between 1 and 100.")

    # Isolate all memory, checkpoints, files and logs before importing runtime modules.
    root = Path(__file__).resolve().parents[2]
    # Paid runs start the durable local collector instead of silently losing spans.
    # Import before task-private path relocation so Phoenix keeps the shared database.
    from phoenix_runtime import PhoenixServerRuntime
    phoenix_runtime = PhoenixServerRuntime()
    phoenix_runtime.start()
    image = args.image or {
        "test_normal": "personalops-appworld-test-normal:0.1.3.post1",
        "test_challenge": "personalops-appworld-test-challenge:0.1.3.post1",
    }.get(args.split, "personalops-appworld:0.1.3.post1")
    if args.output is None:
        out = root / ".agent/appworld-conversations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    else:
        out = args.output.resolve()
        private_root = (root / ".agent").resolve()
        if not out.is_relative_to(private_root):
            parser.error("--output must stay under the repository .agent directory")
    out.mkdir(parents=True, exist_ok=False)
    import path
    path.AGENT_DATA_ROOT = out / "state"
    path.WORKSPACE_ROOT = out / "workspace"
    path.AGENT_DATA_ROOT.mkdir()
    path.WORKSPACE_ROOT.mkdir()
    os.environ.update(EMAIL_MCP_ENABLED="false", MEMORY_ROUTER_ENABLED="false",
                      MEMORY_EXTRACTION_ENABLED="false", MEMORY_WRITE_GATE_ENABLED="false")
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(out / "runtime.log", encoding="utf-8")])

    from config import load_settings
    from evals.appworld.conversation import AppWorldConversation
    from evals.appworld.protocol import DockerWorld
    from evals.appworld.adapter import UsageMeter
    from trace_presentation import attach_audit_callbacks, run_name
    from observability import setup_observability
    import conversation_runtime as cr

    meter = UsageMeter(max_calls=args.max_calls)
    original_factory = cr.build_role_model
    # Process-local wrapper; preserve the project's provider choice and callbacks.
    cr.build_role_model = lambda settings, role: attach_audit_callbacks(original_factory(settings, role), meter)
    os.environ["PHOENIX_PROJECT"] = args.phoenix_project or run_name("AppWorld Task")
    provider = setup_observability()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    trace_exporter = InMemorySpanExporter()
    if provider is not None:
        provider.add_span_processor(SimpleSpanProcessor(trace_exporter))
    from cost_accounting import load_pricing, summarize_cost
    from opentelemetry.sdk.trace import SpanProcessor
    from threading import Lock
    cost_rows = {}
    cost_lock = Lock()
    class CostCapture(SpanProcessor):
        def on_end(self, span):
            a = span.attributes
            if a.get("openinference.span.kind") != "LLM": return
            row = {"request_id":a.get("runtime.request_id"), "role":a.get("runtime.model_role", "unknown"),
                   "model":a.get("llm.model_name"), "cost":json.loads(a.get("cost.details_json", "{}"))}
            with cost_lock: cost_rows[format(span.context.span_id,"016x")] = row
    if provider is not None: provider.add_span_processor(CostCapture())
    try:
        (out / "pricing-snapshot.json").write_text(json.dumps(load_pricing(),ensure_ascii=False,indent=2),encoding="utf-8")
    except Exception as error:
        logging.warning("Pricing snapshot unavailable: %s",type(error).__name__)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    async def run():
        runtime = None
        result = {"task_id": args.task, "status": "starting", "official_evaluation": None}
        save("manifest.json", {"task_id": args.task, "split": args.split,
             "max_calls": args.max_calls, "max_interactions": args.max_interactions,
             "entrypoint": "normal ConversationRuntime", "project": os.environ["PHOENIX_PROJECT"],
             "image": image, "execution_environment": EXECUTION_ENVIRONMENT})
        try:
            with DockerWorld(image=image) as world:
                task = world.request("initialize", split=args.split, task_id=args.task,
                                     trial_id=out.name, max_interactions=args.max_interactions)
                save("task.private.json", task)
                save("isolation.json", world.isolation_report())
                runtime = AppWorldConversation(load_settings(), world, task)
                await runtime.start()
                result.update(await runtime.run_task())
                result["status"] = "finished"
                if result["official_evaluation"] is not None:
                    archive = world.request("export")
                    (out / "execution.private.zip").write_bytes(base64.b64decode(archive["zip_base64"]))
        except Exception as error:
            result.update(status="error", error=repr(error))
            logging.exception("AppWorld task failed; no automatic retry")
            raise
        finally:
            save("result.private.json", result)
            save("usage.private.json", meter.records)
            try:
                if runtime is not None:
                    save("world-calls.private.json", runtime.task_tools.calls)
                    await runtime.stop()
            finally:
                if provider is not None:
                    provider.force_flush(timeout_millis=15000)
                trace_rows = [{
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
                    "status": span.status.status_code.name,
                    "status_description": span.status.description,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "attributes": dict(span.attributes),
                    "events": [
                        {"name": event.name, "attributes": dict(event.attributes)}
                        for event in span.events
                    ],
                } for span in trace_exporter.get_finished_spans()]
                save("spans.private.json", trace_rows)
                with cost_lock:
                    costs = list(cost_rows.values())
                save("cost-summary.json", {"total":summarize_cost([r["cost"] for r in costs]),
                     "by_role":{role:summarize_cost([r["cost"] for r in costs if r["role"]==role]) for role in {r["role"] for r in costs}},
                     "requests":costs, "billing_status":"public_price_estimate_not_invoice"})
                if args.monitor:
                    try:
                        from evals.appworld.batch_monitor import record_trial
                        batch_id = args.monitor_batch or datetime.now(timezone.utc).strftime("adhoc-%Y%m%d")
                        monitor_path = record_trial(out, batch_id)
                        logging.info("AppWorld batch monitor updated: %s", monitor_path)
                    except Exception:
                        logging.exception("AppWorld batch monitor failed; run result remains authoritative")
                print(str(out), flush=True)

    asyncio.run(run())


if __name__ == "__main__":
    main()
