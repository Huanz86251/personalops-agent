"""One explicitly armed real PersonalOps conversation, with private durable evidence.

No batch loop or retry of the conversation. Uses the production ConversationRuntime.
Default mode only starts dependencies and runs free preflight checks.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK = """请完成一次资料核对后编程交付的组合任务：
先用 WEB 分别核对 Python 3.12 官方 csv.DictReader 文档中缺列、多列的行为，以及官方 json 文档中 allow_nan=False 对 NaN/Infinity 的行为。这两个资料方向相互独立，适合在同一个 WEB 步骤中并行核查；只有两个方向，不要扩展研究范围。
优先读 https://docs.python.org/3.12/library/csv.html 和 https://docs.python.org/3.12/library/json.html 。如果文本读取被阻断或是空壳，尝试浏览器或对应官方语言页；记录实际结果，无法核实时明确未知，不编造。
将查证后的简短来源、URL、关键行为、读取限制写成一个或多个资料文件，经过审核形成 INTERNAL_HANDOFF，交给后续 CODE。
然后由 CODE 使用这些资料做一个离线 Python CSV->严格 JSON 转换器，ARTIFACT 交付：convert.py、test_convert.py、README.md，以及一个小的 sample.csv 与转换后的 sample.json。输入列必须恰好为 name,amount；name非空；amount可转为有限浮点数。正常行转换成含 name 字符串与 amount 数字的数组，UTF-8且ensure_ascii=False、allow_nan=False；缺列、多列、NaN、Infinity、空name须清晰报错且非零退出，不能静默丢数据。
CLI格式 python convert.py input.csv output.json；提供至少正常中文行、缺列、多列、NaN、Infinity、空name六类自动测试，并由独立 Code Reviewer 运行核验。数据是你自己构造的合成样例，无需下载真实用户数据。最终回答用中文给出查证结论、文件位置和实测测试结果，任何未完成项要如实说明。
只执行本任务，不发消息、不接邮箱、不创建提醒、不安装依赖、不发布外网。"""


def safe(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


async def run(args):
    from trace_presentation import run_name, attach_audit_callbacks
    project_name = run_name("Chain Audit")
    root_name = run_name("Chain Task")
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "PHOENIX_TRACING_ENABLED": "true", "PHOENIX_TRACE_PROFILE": "curated",
        "PHOENIX_HOST": "127.0.0.1", "PHOENIX_PORT": "6007",
        "PHOENIX_UI_URL": "http://127.0.0.1:6007",
        "PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:6007/v1/traces",
        "PHOENIX_PROJECT_NAME": project_name, "PHOENIX_PROJECT": project_name,
        "PHOENIX_TELEMETRY_ENABLED": "false", "PHOENIX_DISABLE_AGENT_ASSISTANT": "true",
        "PHOENIX_ALLOWED_SANDBOX_PROVIDERS": "NONE", "EMAIL_MCP_ENABLED": "false",
        "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
    })
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(folder / "runtime.log", encoding="utf-8")])
    # Isolate persistent state before importing runtime modules. Reuse model caches,
    # code image and source configuration, never the user's conversations/accounts.
    import path
    original_data = path.AGENT_DATA_ROOT
    from phoenix_runtime import PhoenixServerRuntime
    phoenix = PhoenixServerRuntime()
    path.AGENT_DATA_ROOT = folder / "state"
    path.WORKSPACE_ROOT = folder / "workspace"
    path.AGENT_DATA_ROOT.mkdir(exist_ok=True)
    path.WORKSPACE_ROOT.mkdir(exist_ok=True)
    from config import load_settings
    from observability import setup_observability, trace_span, set_span_output, set_span_attributes, get_tracer
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from langchain_core.callbacks import BaseCallbackHandler
    from evals.appworld.adapter import UsageMeter, _trace_messages
    from evals.appworld.trace_delivery import wait_for_root
    from skill_runtime import load_catalog

    lock = threading.RLock()
    def save(name, value):
        with lock:
            (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=safe), encoding="utf-8")

    def append(value):
        with lock, (folder / "events.private.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **value}, ensure_ascii=False, default=safe) + "\n")
            stream.flush()

    class Meter(UsageMeter):
        failed = False
        def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
            if str(run_id) in self.started:
                return
            if not args.execute_once:
                from trace_callbacks import CALLBACK
                CALLBACK.reject_before_send(run_id, "Preflight forbids paid model calls")
                raise RuntimeError("Preflight forbids paid model calls")
            if self.failed:
                from trace_callbacks import CALLBACK
                CALLBACK.reject_before_send(run_id, "Audit stopped after model error")
                raise RuntimeError("Audit stopped after model error; further paid calls forbidden")
            super().on_chat_model_start(serialized, messages, run_id=run_id, **kw)
            key = str(run_id)
            metadata = kw.get("metadata") or {}
            self.call_metadata[key]["runtime_metadata"] = metadata
            self.call_metadata[key]["parent_span_id"] = format(trace.get_current_span().get_span_context().span_id, "016x")
            append({"event": "model_start", "id": key, "metadata": self.call_metadata[key], "messages": _trace_messages(messages)})
            save("usage.json", self.report())

        def on_llm_end(self, response, *, run_id, **kw):
            super().on_llm_end(response, run_id=run_id, **kw)
            append({"event": "model_end", "id": str(run_id), "record": self.records.get(str(run_id)), "response": response})
            save("usage.json", self.report())

        def on_llm_error(self, error, *, run_id, **kw):
            self.failed = True
            super().on_llm_error(error, run_id=run_id, **kw)
            append({"event": "model_error", "id": str(run_id), "error": repr(error)})
            save("usage.json", self.report())

    class ToolLedger(BaseCallbackHandler):
        def __init__(self):
            self.spans = {}
        def on_tool_start(self, serialized, input_str, *, run_id, **kw):
            key = str(run_id)
            if key in self.spans:
                return
            name = (serialized or {}).get("name", "unknown")
            if (kw.get("metadata") or {}).get("trace.owner") == "personalops":
                self.spans[key] = None
                append({"event": "tool_start", "id": key, "name": name, "input": kw.get("inputs") or input_str})
                return
            span = get_tracer().start_span("TOOL / " + name, openinference_span_kind="tool")
            span.set_input(kw.get("inputs") or input_str)
            set_span_attributes(span, **{"tool.name": name, "audit.tool_run_id": key})
            self.spans[key] = span
            append({"event": "tool_start", "id": key, "name": name, "input": kw.get("inputs") or input_str})
        def on_tool_end(self, output, *, run_id, **kw):
            span = self.spans.pop(str(run_id), None)
            if span is None:
                append({"event": "tool_end", "id": str(run_id), "failed": getattr(output, "status", None) == "error", "output": output})
                return
            failed = getattr(output, "status", None) == "error"
            span.set_output(safe(output))
            span.set_status(Status(StatusCode.ERROR if failed else StatusCode.OK))
            span.end()
            append({"event": "tool_end", "id": str(run_id), "failed": failed, "output": output})
        def on_tool_error(self, error, *, run_id, **kw):
            span = self.spans.pop(str(run_id), None)
            if span is not None:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                span.end()
            append({"event": "tool_error", "id": str(run_id), "error": repr(error)})

    meter = Meter(max_calls=80)
    ledger = ToolLedger()
    import conversation_runtime as cr
    # Instrument model construction before startup; no provider/framework duplicate
    # instrumentation. Model-bound callbacks also cover title and background calls.
    original_role_factory = cr.build_role_model
    def factory(settings, role):
        # Disable retries before SDK clients are constructed, for every role.
        role_settings = replace(settings.role_models[role], max_retries=0)
        scoped = replace(settings, role_models={**settings.role_models, role: role_settings})
        model = original_role_factory(scoped, role)
        attach_audit_callbacks(model, meter)
        model.tags = [*(model.tags or []), "appworld:" + role]
        return model
    cr.build_role_model = factory
    from tools import ALL_TOOLS
    settings = load_settings()
    settings = replace(settings, code_sandbox=replace(settings.code_sandbox, auto_build=False))
    runtime = cr.ConversationRuntime(settings, ALL_TOOLS)
    result = {"task": TASK, "execute_once": args.execute_once, "output": str(folder)}
    save("task.json", result)
    save("catalog.private.json", load_catalog())
    save("configuration.json", {"planning": asdict(settings.planning), "worker_runtime": asdict(settings.worker_runtime), "concurrency": asdict(settings.runtime_concurrency), "sandbox": asdict(settings.code_sandbox), "model": settings.llm_model, "hard_model": settings.hard_llm_model, "max_output_tokens": settings.cloud_llm_max_tokens, "scheduler_thinking_enabled": settings.scheduler_thinking_enabled, "llm_thinking_enabled": settings.llm_thinking_enabled, "max_calls": meter.max_calls, "transport_retries": 0, "original_data_untouched": str(original_data)})
    save("source-hashes.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ("*.py", "skills/**/SKILL.md", "prompts/**/*.md") for p in ROOT.glob(pattern)})
    provider = None
    try:
        await asyncio.to_thread(phoenix.start)
        provider = setup_observability()
        if provider is None:
            raise RuntimeError("Trace collector required before paid execution")
        with trace_span("Delivery Check", kind="chain") as probe:
            probe_id = format(probe.get_span_context().trace_id, "032x")
        provider.force_flush(timeout_millis=10000)
        delivery = await asyncio.to_thread(wait_for_root, phoenix.database_path, probe_id, "Delivery Check")
        save("collector-preflight.json", {"trace_id": probe_id, **delivery})
        if not delivery["root_persisted"]:
            raise RuntimeError("Preflight trace did not reach Phoenix SQLite; paid execution blocked")
        with trace_span(root_name, kind="chain", input_value=result) as root:
            result["trace_id"] = format(root.get_span_context().trace_id, "032x")
            save("result.json", result)
            with trace_span("PREPARE / production runtime", kind="chain"):
                await runtime.start()
                import numpy as np
                vectors = await runtime.retrieval_models.aembed_documents(["本地模型推理预检", "长文本边界" * 300])
                if not np.isfinite(vectors).all():
                    raise RuntimeError("Embedding preflight returned non-finite values")
                ranked = await runtime.retrieval_models.arerank(query="查阅Python官方文档", documents=["Web公开技术文档检索", "发送邮件"], top_k=2)
                if not ranked or not all(np.isfinite(x.score) for x in ranked):
                    raise RuntimeError("Reranker preflight failed")
                save("local-inference-preflight.json", {"embedding_shape": list(np.asarray(vectors).shape), "finite": True, "reranking": [safe(x) for x in ranked]})
                await asyncio.to_thread(runtime.code_runtime.sandbox_manager.ensure_ready)
                async with runtime.playwright_mcp.lease("chain-preflight") as browser_runtime:
                    save("browser-tools.json", [t.name for t in browser_runtime.tools])
                runtime.planning_graph = runtime.planning_graph.with_config(callbacks=[ledger])
                result["preflight"] = "runtime_ready"
                save("result.json", result)
            if args.execute_once:
                # Exclusive marker makes reusing this evidence directory fail closed.
                with (folder / "PAID_RUN_STARTED").open("x", encoding="utf-8") as f:
                    f.write(datetime.now(timezone.utc).isoformat())
                async def progress(event):
                    append({"event": "progress", "value": event})
                result["answer"] = await runtime.ask(user_text=TASK, channel="chain-audit", external_chat_id="isolated", event_id="chain_audit_20260906", progress_callback=progress)
                (folder / "answer.md").write_text(result["answer"], encoding="utf-8")
                result["status"] = "returned"
            else:
                result["status"] = "preflight_only"
            await runtime.stop()
            result["usage"] = meter.report()
            set_span_output(root, result)
        provider.force_flush(timeout_millis=10000)
        result["trace_delivery"] = await asyncio.to_thread(wait_for_root, phoenix.database_path, result["trace_id"], root_name)
    except BaseException as error:
        result["status"] = "exception"
        result["error"] = repr(error)
        (folder / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            await runtime.stop()
        finally:
            result["usage"] = meter.report()
            save("result.json", result)
            save("usage.json", meter.report())
            if provider:
                provider.force_flush(timeout_millis=10000)
            # Leave the local UI available for inspection; do not stop a reused server.
            append({"event": "finished", "status": result.get("status")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-once", action="store_true")
    parser.add_argument("--output", required=True)
    asyncio.run(run(parser.parse_args()))
