"""Short run names and honest elapsed-time measurements."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from time import perf_counter
from threading import RLock
from uuid import uuid4

PARENT = ContextVar("trace_timing_parent", default=None)

def run_name(purpose):
    return f"{purpose} · {datetime.now().astimezone():%m-%d %H:%M:%S} · {uuid4().hex[:4]}"

def role_badge(name):
    for role, badge in (("Code Reviewer", "🔍"), ("Code Worker", "🛠️"), ("Code Agent", "🧩"),
                        ("Web Agent", "🌐"), ("General Agent", "💬")):
        if name.startswith(role):
            return f"{badge} {name}"
    return name

@contextmanager
def timing_scope(span):
    start = perf_counter()
    parent = PARENT.get()
    current = {"start": start, "last_end": None, "active": 0, "lock": RLock()}
    span.set_attribute("timing.started_at", datetime.now().astimezone().isoformat(timespec="milliseconds"))
    if parent:
        with parent["lock"]:
            span.set_attribute("timing.parent_offset_ms", (start - parent["start"]) * 1000)
            overlap = parent["active"] > 0
            span.set_attribute("timing.overlaps_sibling", overlap)
            if not overlap and parent["last_end"] is not None:
                span.set_attribute("timing.previous_sibling_gap_ms", (start - parent["last_end"]) * 1000)
            parent["active"] += 1
    token = PARENT.set(current)
    try:
        yield
    finally:
        end = perf_counter()
        PARENT.reset(token)
        elapsed = (end-start)*1000
        span.set_attribute("timing.duration_ms", elapsed)
        span.add_event("Elapsed time", {"timing.duration_ms": elapsed})
        if parent:
            with parent["lock"]:
                parent["active"] -= 1
                parent["last_end"] = end

def attach_audit_callbacks(model, meter):
    """Budget gate first; retain canonical tracing and existing callbacks."""
    from trace_callbacks import callbacks
    existing = getattr(model, "callbacks", None)
    existing = list(getattr(existing, "handlers", existing) or [])
    model.callbacks = [meter, *[cb for cb in callbacks(existing) if cb is not meter]]
    return model
