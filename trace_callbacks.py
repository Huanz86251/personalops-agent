"""Single owner for LLM/tool leaf spans, independent of provider SDK tracing."""
from threading import RLock
from time import perf_counter
from langchain_core.callbacks import BaseCallbackHandler
from observability import get_tracer, set_span_input, set_span_output, set_span_attributes
from runtime_tracing import ROLE_NAMES, TRACE_IDS
from usage_accounting import SCOPES, normalize_usage, FIELDS


class RuntimeTraceCallback(BaseCallbackHandler):
    run_inline = True

    def __init__(self):
        self.spans = {}
        self.finished = set()
        self.usage_context = {}
        self.clocks = {}
        self.lock = RLock()
        self.display_counts = {}
        self.chat_inputs = {}

    def _start(self, run_id, name, kind, inputs, attrs):
        key = str(run_id)
        with self.lock:
            if key in self.spans or key in self.finished:
                return
            tracer = get_tracer()
            if tracer is None:
                return
            span = tracer.start_span(name, openinference_span_kind=kind)
            # Number repeated calls under their actual trace and parent role.
            context = span.get_span_context()
            parent = getattr(span, "parent", None)
            counter_key = (context.trace_id, getattr(parent, "span_id", 0), name)
            self.display_counts[counter_key] = self.display_counts.get(counter_key, 0) + 1
            ordinal = self.display_counts[counter_key]
            span.update_name(f"{name} · {ordinal:02d}")
            span.set_attribute("runtime.display_ordinal", ordinal)
            if len(self.display_counts) > 10000:
                self.display_counts = {counter_key: ordinal}
            self.spans[key] = span
            self.clocks[key] = {"start": perf_counter(), "first": None, "kind": kind}
            if kind == "llm":
                self.chat_inputs[key] = inputs
                self.usage_context[key] = (SCOPES.get(), attrs.get("runtime.model_role", "model"),
                    {**TRACE_IDS.get(), **{k.removeprefix("runtime."): v for k, v in attrs.items() if k.startswith("runtime.") and k != "runtime.model_role"}})
        set_span_input(span, inputs)
        set_span_attributes(span, **{"runtime.request_id": key, "trace.owner": "personalops",
            **{"runtime." + k: v for k, v in TRACE_IDS.get().items()}, **attrs})

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        metadata = kwargs.get("metadata") or {}
        role = metadata.get("runtime.model_role", "model")
        params = kwargs.get("invocation_params") or {}
        model = metadata.get("ls_model_name") or params.get("model") or params.get("model_name") or "unknown"
        # Do not serialize the client or invocation parameters: they can carry credentials.
        from trace_chat import message_record
        request_options = {k: params[k] for k in ("tools", "tool_choice", "response_format") if k in params}
        extra = params.get("extra_body") or {}
        # Only retain allowlisted public controls, never arbitrary provider extras.
        thinking_options = {k: extra[k] for k in ("enable_thinking", "reasoning_effort", "thinking_budget", "preserve_thinking") if k in extra}
        if thinking_options:
            request_options["thinking"] = thinking_options
        import sys, json
        evidence_module = sys.modules.get('workers.evidence_refs')
        evidence_refs = evidence_module.ACTIVE_EVIDENCE_REFS.get() if evidence_module else None
        evidence_attrs = ({'evidence.references_json': json.dumps(
            {short: raw for raw, short in evidence_refs.items()}, ensure_ascii=False)} if evidence_refs else {})
        self._start(run_id, "LLM / " + ROLE_NAMES.get(role, role), "llm",
            {"messages": [[message_record(m) for m in batch] for batch in messages],
             "request_options": request_options},
            {"llm.model_name": str(model), "runtime.model_role": role, "usage.status": "pending",
             **evidence_attrs,
             **{k: v for k, v in metadata.items() if k.startswith("runtime.")}})

    def _finish(self, run_id, output, attrs=None, error=None, usage=None):
        key = str(run_id)
        with self.lock:
            span = self.spans.pop(key, None)
            if span is None:
                return
            chat_input = self.chat_inputs.pop(key, None)
            usage_context = self.usage_context.pop(key, None)
            clock = self.clocks.pop(key)
            self.finished.add(key)
            # Bounded dedupe window; active spans are never evicted.
            if len(self.finished) > 10000:
                self.finished = {key}
        try:
            span.set_attribute("timing.duration_ms", (perf_counter() - clock["start"]) * 1000)
            if clock["kind"] == "llm":
                span.set_attribute("timing.first_token_status", "observed" if clock["first"] is not None else "unavailable")
                if clock["first"] is not None:
                    span.set_attribute("timing.first_token_ms", (clock["first"]-clock["start"]) * 1000)
            if usage_context:
                scopes, role, ids = usage_context
                for ledger in scopes:
                    ledger.record(key, role, ids, usage or dict.fromkeys(FIELDS))
            if chat_input is not None:
                try:
                    from trace_chat import emit_chat_view, chat_attributes
                    set_span_attributes(span, **{k:v for k,v in chat_attributes(chat_input, output).items() if k.startswith("llm.")})
                    emit_chat_view(get_tracer(), span, chat_input, output)
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Chat trace presentation failed")
            set_span_output(span, output)
            set_span_attributes(span, **(attrs or {}))
            if (attrs or {}).get("usage.status") == "missing":
                span.add_event("Usage unavailable", {"usage.note": "Token usage was not reported; do not interpret as measured zero."})
            from opentelemetry.trace import Status, StatusCode
            if error is not None:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
            else:
                span.set_status(Status(StatusCode.OK))
        finally:
            span.end()

    def on_llm_end(self, response, *, run_id, **kwargs):
        messages = [g.message for batch in response.generations for g in batch if hasattr(g, "message")]
        usage = normalize_usage(response, messages)
        attrs = {"usage.status": "reported" if usage["total_tokens"] is not None else "missing"}
        if any(v is not None for v in usage.values()):
            for source, target in (("input_tokens", "prompt"), ("output_tokens", "completion"), ("total_tokens", "total")):
                if isinstance(usage.get(source), int):
                    attrs["llm.token_count." + target] = usage[source]
            for key in ("cache_read_tokens", "cache_creation_tokens", "reasoning_tokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    attrs["usage." + key] = value
        from cost_accounting import estimate_cost, cost_attributes
        active = self.spans.get(str(run_id))
        model_name = (getattr(active, "attributes", {}) or {}).get("llm.model_name", "unknown")
        cost = estimate_cost(model_name, usage)
        attrs.update(cost_attributes(cost))
        usage["cost"] = cost
        from trace_chat import message_record
        self._finish(run_id, {"responses": [{**message_record(m), "response_id": getattr(m, "id", None)} for m in messages],
                              "usage": usage, "llm_output": response.llm_output}, attrs, usage=usage)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, {"error": str(error)}, {"usage.status": "missing"}, error)

    def reject_before_send(self, run_id, reason):
        """Close an inherited start callback without inventing a provider call."""
        key = str(run_id)
        with self.lock:
            if key in self.finished:
                return
            span = self.spans.pop(key, None)
            self.usage_context.pop(key, None)
            self.chat_inputs.pop(key, None)
            clock = self.clocks.pop(key, None)
            self.finished.add(key)
            if len(self.finished) > 10000:
                self.finished = {key}
        if span is None:
            tracer = get_tracer()
            if tracer is None:
                return
            span = tracer.start_span("Request / Rejected", openinference_span_kind="chain")
        try:
            span.update_name("Request / Rejected")
            set_span_attributes(span, **{"openinference.span.kind":"CHAIN", "request.sent":False,
                "runtime.request_id":key, "business.status":"REJECTED", "usage.status":"not_sent"})
            if clock:
                span.set_attribute("timing.duration_ms", (perf_counter()-clock["start"])*1000)
            set_span_output(span, {"reason":str(reason), "request_sent":False,
                "usage_note":"Rejected before provider invocation; excluded from token usage totals."})
            from opentelemetry.trace import Status, StatusCode
            span.set_status(Status(StatusCode.ERROR, str(reason)))
        finally:
            span.end()

    def on_llm_new_token(self, token, *, run_id, **kwargs):
        if not token:
            return
        with self.lock:
            clock = self.clocks.get(str(run_id))
            if clock is not None and clock["first"] is None:
                clock["first"] = perf_counter()

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        from trace_overview import tool_purpose
        with self.lock:
            if str(run_id) in self.spans or str(run_id) in self.finished:
                return
        name = (serialized or {}).get("name", "unknown")
        self._start(run_id, tool_purpose(name, kwargs.get("inputs") or input_str), "tool", kwargs.get("inputs") or input_str,
                    {"tool.name": name, **{k: v for k, v in (kwargs.get("metadata") or {}).items() if k.startswith("runtime.")}})
        parent = self.spans.get(str(run_id))
        if parent is not None:
            from trace_chat import emit_code_view
            emit_code_view(get_tracer(), parent, kwargs.get("inputs") or input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        failed = getattr(output, "status", None) == "error"
        content = getattr(output, "content", output)
        if isinstance(content, str):
            import json
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                content = None
        fetch = content.get("fetch_status") if isinstance(content, dict) else None
        attrs = {"business.status": fetch or ("FAILED" if failed else "RETURNED")}
        if fetch:
            attrs.update({"web.fetch_status": fetch, "web.evidence_available": content.get("evidence_available", False)})
            if content.get("http_status") is not None:
                attrs["http.status_code"] = content["http_status"]
        self._finish(run_id, {"content": getattr(output, "content", str(output))},
                     attrs,
                     RuntimeError("Tool returned an error result") if failed else None)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, {"error": str(error)}, error=error)


CALLBACK = RuntimeTraceCallback()

def callbacks(existing=None):
    values = list(existing or [])
    return values if CALLBACK in values else [*values, CALLBACK]

def configure_graph(graph, role):
    """Keep CompiledStateGraph APIs while covering direct and runtime callers."""
    return graph.with_config({"callbacks": callbacks(),
                              "metadata": {"runtime.model_role": role, "trace.owner": "personalops"}})
