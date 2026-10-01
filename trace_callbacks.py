"""Single owner for LLM/tool leaf spans, independent of provider SDK tracing."""
from threading import RLock
from time import perf_counter
from langchain_core.callbacks import BaseCallbackHandler
from observability import get_tracer, set_span_input, set_span_output, set_span_attributes
from runtime_tracing import ROLE_NAMES, TRACE_IDS
from usage_accounting import SCOPES, normalize_usage, FIELDS


def _current_request_usage_attributes(usage):
    """Build non-cumulative usage fields for one provider request.

    Phoenix also displays cumulative token counts on parent spans.  These
    explicitly named attributes make it clear that the values belong only to
    the selected LLM leaf span / provider request.
    """

    incoming = usage.get("input_tokens")
    outgoing = usage.get("output_tokens")
    cached = usage.get("cache_read_tokens")
    attrs = {
        "usage.current.scope": "single_provider_request",
        "usage.current.input_tokens": incoming,
        "usage.current.output_tokens": outgoing,
        "usage.current.cache_read_tokens": cached,
        "usage.current.cache_status": (
            "reported" if isinstance(cached, int) else "unavailable"
        ),
    }
    if isinstance(incoming, int) and isinstance(cached, int):
        uncached = max(incoming - cached, 0)
        ratio = cached / incoming if incoming else None
        attrs["usage.current.uncached_input_tokens"] = uncached
        if ratio is not None:
            attrs["usage.current.cache_hit_ratio"] = ratio
            attrs["usage.current.cache_hit_percent"] = round(ratio * 100, 2)
            attrs["usage.current.summary"] = (
                f"input={incoming} | cache_read={cached} ({ratio * 100:.1f}%) | "
                f"uncached={uncached} | output={outgoing if isinstance(outgoing, int) else 'unknown'}"
            )
    elif isinstance(incoming, int):
        attrs["usage.current.summary"] = (
            f"input={incoming} | cache_read=unavailable | "
            f"output={outgoing if isinstance(outgoing, int) else 'unknown'}"
        )
    return {key: value for key, value in attrs.items() if value is not None}


def _is_graph_interrupt(error):
    """LangGraph用GraphInterrupt表示可恢复暂停，不是业务失败。"""
    try:
        from langgraph.errors import GraphInterrupt
    except ImportError:
        return False
    return isinstance(error, GraphInterrupt)


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
        self.tool_context = {}
        self.paused_progress_gates = {}

    @staticmethod
    def _progress_gate_key(name, parent_run_id, inputs, metadata=None):
        if name != "publish_worker_progress":
            return None
        import json
        try:
            payload = json.dumps(inputs, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            payload = str(inputs)
        metadata = metadata or {}
        worker_identity = (
            metadata.get("runtime.worker_id")
            or metadata.get("worker_id")
            or parent_run_id
            or ""
        )
        return (str(worker_identity), name, payload)

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
            display_name = f"{name} · {ordinal:02d}"
            span.update_name(display_name)
            span.set_attribute("runtime.display_ordinal", ordinal)
            if len(self.display_counts) > 10000:
                self.display_counts = {counter_key: ordinal}
            self.spans[key] = span
            self.clocks[key] = {
                "start": perf_counter(),
                "first": None,
                "kind": kind,
                "display_name": display_name,
            }
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
        from runtime_tool_visibility import VISIBLE_MODEL_TOOLS
        visible_tools = list(VISIBLE_MODEL_TOOLS.get())
        if visible_tools:
            request_options["visible_tools"] = visible_tools
        import sys, json
        evidence_module = sys.modules.get('workers.evidence_refs')
        evidence_refs = evidence_module.ACTIVE_EVIDENCE_REFS.get() if evidence_module else None
        evidence_attrs = ({'evidence.references_json': json.dumps(
            {short: raw for raw, short in evidence_refs.items()}, ensure_ascii=False)} if evidence_refs else {})
        self._start(run_id, "LLM / " + ROLE_NAMES.get(role, role), "llm",
            {"messages": [[message_record(m) for m in batch] for batch in messages],
             "request_options": request_options},
            {"llm.model_name": str(model), "runtime.model_role": role, "usage.status": "pending",
             "tools.visible_count": len(visible_tools),
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
            self.tool_context.pop(key, None)
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
            if clock["kind"] == "llm":
                current = attrs or {}
                incoming = current.get("usage.current.input_tokens")
                outgoing = current.get("usage.current.output_tokens")
                cache_percent = current.get("usage.current.cache_hit_percent")
                if isinstance(incoming, int) and isinstance(outgoing, int):
                    cache_label = (
                        f"{cache_percent:.1f}%"
                        if isinstance(cache_percent, (int, float))
                        else "n/a"
                    )
                    span.update_name(
                        f"{clock['display_name']} · I/O {incoming:,}→{outgoing:,} "
                        f"· Cache {cache_label}"
                    )
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
        current_usage_attrs = _current_request_usage_attributes(usage)
        attrs.update(current_usage_attrs)
        active = self.spans.get(str(run_id))
        if active is not None and current_usage_attrs:
            active.add_event(
                "Current LLM request usage",
                {
                    key: value
                    for key, value in current_usage_attrs.items()
                    if key != "usage.current.scope"
                },
            )
        from cost_accounting import estimate_cost, cost_attributes
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
        name = (serialized or {}).get("name", "unknown")
        inputs = kwargs.get("inputs") or input_str
        pause_key = self._progress_gate_key(
            name,
            kwargs.get("parent_run_id"),
            inputs,
            kwargs.get("metadata"),
        )
        key = str(run_id)
        with self.lock:
            if key in self.spans or key in self.finished:
                return
            paused = self.paused_progress_gates.pop(pause_key, None) if pause_key else None
            if paused is not None:
                span, clock = paused
                self.spans[key] = span
                self.clocks[key] = clock
                self.tool_context[key] = {
                    "name": name,
                    "pause_key": pause_key,
                }
                span.update_name("Worker Progress Gate")
                span.add_event("Worker progress gate resumed")
                set_span_attributes(span, **{
                    "business.status": "RESUMED",
                    "worker.progress_gate.transition": "PAUSED_TO_CONTINUED",
                })
                return
        self._start(run_id, tool_purpose(name, inputs), "tool", inputs,
                    {"tool.name": name, **{k: v for k, v in (kwargs.get("metadata") or {}).items() if k.startswith("runtime.")}})
        with self.lock:
            self.tool_context[key] = {
                "name": name,
                "pause_key": pause_key,
            }
        parent = self.spans.get(key)
        if parent is not None:
            from trace_chat import emit_code_view
            emit_code_view(get_tracer(), parent, inputs)

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
        key = str(run_id)
        if _is_graph_interrupt(error):
            with self.lock:
                context = self.tool_context.pop(key, None) or {}
                pause_key = context.get("pause_key")
                span = self.spans.pop(key, None)
                clock = self.clocks.pop(key, None)
                self.finished.add(key)
                if span is not None and clock is not None and pause_key is not None:
                    span.update_name("Worker Progress Gate")
                    span.add_event("Worker progress gate paused")
                    set_span_attributes(span, **{
                        "business.status": "PAUSED",
                        "worker.progress_gate.transition": "PAUSED",
                        "control_flow.expected_interrupt": True,
                    })
                    self.paused_progress_gates[pause_key] = (span, clock)
                    return
        self._finish(run_id, {"error": str(error)}, error=error)


CALLBACK = RuntimeTraceCallback()

def callbacks(existing=None):
    values = list(existing or [])
    return values if CALLBACK in values else [*values, CALLBACK]

def configure_graph(graph, role):
    """Keep CompiledStateGraph APIs while covering direct and runtime callers."""
    return graph.with_config({"callbacks": callbacks(),
                              "metadata": {"runtime.model_role": role, "trace.owner": "personalops"}})
