"""Request-based usage rollups; never sum parent and child token attributes."""
from contextvars import ContextVar
from contextlib import contextmanager
from threading import RLock
import json

SCOPES = ContextVar("usage_scopes", default=())
FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_creation_tokens", "reasoning_tokens")
SCHEDULER_ROLES = {"scheduler", "code_scheduler", "replanner", "final_reviewer", "worker_leader", "reporter", "web_reporter"}


def normalize_usage(response, messages):
    """Prefer normalized SDK usage; fall back to provider response metadata."""
    raw = next((m.usage_metadata for m in messages if getattr(m, "usage_metadata", None)), None)
    if raw is None:
        raw = next(((getattr(m, "response_metadata", {}) or {}).get("token_usage") or
                    (getattr(m, "response_metadata", {}) or {}).get("usage") for m in messages
                    if (getattr(m, "response_metadata", {}) or {}).get("token_usage") or
                    (getattr(m, "response_metadata", {}) or {}).get("usage")), None)
    raw = raw or (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    def number(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    result = {"input_tokens": number(raw.get("input_tokens", raw.get("prompt_tokens"))),
              "output_tokens": number(raw.get("output_tokens", raw.get("completion_tokens"))),
              "total_tokens": number(raw.get("total_tokens"))}
    if result["total_tokens"] is None and all(result[k] is not None for k in ("input_tokens", "output_tokens")):
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    incoming = raw.get("input_token_details") or raw.get("prompt_tokens_details") or {}
    outgoing = raw.get("output_token_details") or raw.get("completion_tokens_details") or {}
    result.update(cache_read_tokens=number(incoming.get("cache_read", incoming.get("cached_tokens", raw.get("prompt_cache_hit_tokens")))),
                  cache_creation_tokens=number(incoming.get("cache_creation", raw.get("cache_creation_input_tokens"))),
                  reasoning_tokens=number(outgoing.get("reasoning", outgoing.get("reasoning_tokens"))))
    return result


def totals(rows):
    result = {"requests": len(rows)}
    for field in FIELDS:
        values = [row["usage"].get(field) for row in rows]
        known = [value for value in values if value is not None]
        result[field] = sum(known) if known else None
        result[field + "_missing_requests"] = len(values) - len(known)
    result["status"] = "complete" if rows and not result["total_tokens_missing_requests"] else "partial" if any(r["usage"].get("total_tokens") is not None for r in rows) else "unknown" if rows else "no_requests"
    result["cache_hit_ratio"] = (result["cache_read_tokens"] / result["input_tokens"]
        if result["input_tokens"] and not result["cache_read_tokens_missing_requests"] and not result["input_tokens_missing_requests"] else None)
    from cost_accounting import summarize_cost
    result["cost"] = summarize_cost([row["usage"].get("cost", {}) for row in rows])
    result["cost_status"] = result["cost"]["status"]
    return result


class Ledger:
    def __init__(self):
        self.rows = {}
        self.lock = RLock()

    def record(self, key, role, ids, usage):
        with self.lock:
            self.rows.setdefault(key, {"request_id": key, "role": role, **ids, "usage": usage})

    def report(self):
        with self.lock:
            rows = list(self.rows.values())
        groups = {}
        for field in ("role", "step_id", "candidate_revision", "repair_round"):
            groups[field] = {str(value): totals([r for r in rows if r.get(field) == value])
                             for value in dict.fromkeys(r.get(field) for r in rows) if value is not None}
        rounds = {}
        for row in rows:
            if row["role"] in {"reviewer", "code_reviewer"}:
                key = "/".join(str(row.get(k, "unknown")) for k in ("step_id", "attempt_id", "candidate_revision", "repair_round"))
                rounds.setdefault(key, []).append(row)
        groups["code_review_round"] = {key: totals(values) for key, values in rounds.items()}
        cumulative = 0
        missing = 0
        requests = []
        for row in rows:
            value = row["usage"].get("total_tokens")
            cumulative += value or 0
            missing += value is None
            requests.append({**row, "cumulative_known_total_tokens": cumulative,
                             "cumulative_missing_requests": missing})
        return {"subtree": totals(rows), "scheduler_control": totals([r for r in rows if r["role"] in SCHEDULER_ROLES]),
                "code_review": totals([r for r in rows if r["role"] in {"reviewer", "code_reviewer"}]),
                "by": groups, "requests_in_completion_order": requests,
                "note": "Cache is part of input; reasoning is part of output. Parent rollups overlap: do not add them. Missing values are unknown, not zero."}


@contextmanager
def usage_scope(span):
    ledger = Ledger()
    token = SCOPES.set((*SCOPES.get(), ledger))
    try:
        yield ledger
    finally:
        SCOPES.reset(token)
        report = ledger.report()
        if report["subtree"]["requests"]:
            # Custom attributes avoid Phoenix summing rollups with canonical LLM leaves.
            span.set_attribute("usage.summary_json", json.dumps(report, ensure_ascii=False))
            span.set_attribute("usage.rollup.status", report["subtree"]["status"])
            cost = report["subtree"]["cost"]
            span.set_attribute("cost.rollup_json", json.dumps(cost))
            if cost["known_total_cny"] is not None:
                span.set_attribute("cost.rollup.known_total_cny", cost["known_total_cny"])
            for field in FIELDS:
                value = report["subtree"][field]
                if value is not None:
                    span.set_attribute("usage.rollup." + field, value)
            span.add_event("Token usage summary", {"usage.summary": json.dumps({"subtree": report["subtree"], "scheduler_control": report["scheduler_control"]})})
