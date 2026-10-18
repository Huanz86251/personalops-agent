"""Portable worker-authored knowledge; preserved without a second summarization."""
from pydantic import Field
from schemas.handoff import HandoffKnowledge


def knowledge_field():
    return Field(default_factory=list, max_length=12, description=(
        "Knowledge handoff to a NEW agent that does not automatically receive your conversation, tool outputs, "
        "or private variables; registered execution history may be read on demand. Include all task-relevant discoveries needed to continue without "
        "repeating exploration: exact API usage and return structure, parameter sources, useful "
        "code/web findings, known pitfalls and remaining work. A name alone is insufficient. "
        "Use [] only when there is no reusable knowledge. These are worker-authored claims, "
        "not an independent verification verdict. Cite registered evidence_refs for factual discoveries, API docs and observed execution; suggestions may use []. Harness only resolves references, not the meaning of your claims."
    ))


def check_handoff_knowledge(items, state):
    """Resolve citations independently; never validate claim semantics or rerun tools."""
    from workers.submission import resolve_tool_evidence
    from workers.evidence_refs import ACTIVE_EVIDENCE_REFS, CONTROL
    refs = state.get("worker_evidence_refs") or ACTIVE_EVIDENCE_REFS.get() or {}
    inverse = {short: raw for raw, short in refs.items()}
    messages = [*state.get("worker_archived_messages", []),
                *state.get("handoff_source_messages", []), *state.get("messages", [])]
    checked = []
    for raw in items:
        item = HandoffKnowledge.model_validate(raw)
        resolved = []; unresolved = []; errors = {}; canonical = []
        for ref in dict.fromkeys(item.evidence_refs):
            call_id = inverse.get(ref, ref)
            try:
                evidence = resolve_tool_evidence(messages, [call_id])[0]
                if evidence.tool_name in CONTROL:
                    raise ValueError("control tool cannot support a knowledge handoff")
                # Keep failed tool returns too: they support failure, not success.
                envelope = evidence.model_dump(mode="json")
                result_message = next((m for m in reversed(messages)
                    if (m.get("tool_call_id") if isinstance(m, dict) else getattr(m, "tool_call_id", None)) == call_id), None)
                envelope["result_status"] = (result_message.get("status", "unknown") if isinstance(result_message, dict)
                    else getattr(result_message, "status", "unknown"))
                envelope["origin_worker"] = state.get("worker_id")
                envelope["origin_step"] = state.get("step_id")
                resolved.append(envelope)
                canonical.append(call_id)
            except (ValueError, TypeError, KeyError) as error:
                unresolved.append(ref); errors[ref] = str(error)
        checked.append(item.model_copy(update={
            "evidence_refs": canonical,
            "resolved_evidence": resolved,
            "unresolved_evidence_refs": unresolved,
            "evidence_errors": errors,
            "evidence_check": "PARTIAL" if resolved and unresolved else "RESOLVED" if resolved else "UNSUPPORTED",
        }))
    return checked


def collect_handoff_knowledge(traces):
    """Take accepted submissions only, never model-generated reporter replacements.

    Later corrections of the same topic/source supersede older ones. A local
    evidence alias is deliberately not transported as a globally valid identity.
    """
    selected = {}
    for trace in traces:
        for key in ("worker_submission", "code_worker_submission", "general_result"):
            value = trace.get(key) or {}
            if not isinstance(value, dict):
                continue
            value = value.get("submission", value)
            for raw in value.get("handoff_knowledge", []):
                item = HandoffKnowledge.model_validate(raw)
                # Accepted submissions carry Harness-resolved durable envelopes.
                # Older traces are checked against actual request/result records.
                if isinstance(raw, dict) and "evidence_check" not in raw:
                    item = check_handoff_knowledge([item], trace)[0]
                selected[(item.topic, item.source)] = item
    return list(selected.values())

def compact_direct_handoff(reports, max_chars=1800):
    """Keep the newest Worker handoff whole; bound only older handoffs.

    The newest report is the next Worker's primary handoff.  None of its valid
    evidence associations, failure details, or worker notes may be omitted by the legacy
    character budget.  Older reports remain available in execution history and
    are added only when they fit as whole entries.
    """
    import json

    normalized_reports = []
    for report_index, report in enumerate(reports):
        value = report.model_dump(mode="json") if hasattr(report, "model_dump") else report
        if isinstance(value, dict):
            normalized_reports.append((report_index, value))
    if not normalized_reports:
        return {}
    latest_report_index = normalized_reports[-1][0]

    knowledge = {}
    review_outcomes = {}

    def text(value, limit, *, latest):
        normalized = str(value or "")
        return normalized if latest else normalized[:limit]

    for report_index, value in normalized_reports:
        latest = report_index == latest_report_index
        status = str(value.get("status") or "")
        errors = list(value.get("errors", []) or [])
        unresolved = list(value.get("unresolved_items", []) or [])
        if status and (status != "COMPLETED" or errors or unresolved):
            key = str(value.get("step_id") or report_index + 1)
            review_outcomes[key] = (
                report_index,
                {
                    "step_id": value.get("step_id"),
                    "status": status,
                    "assessment_source": value.get("assessment_source"),
                    "stop_reason": text(value.get("stop_reason"), 220, latest=latest),
                    "unresolved_items": [
                        text(item, 180, latest=latest)
                        for item in (unresolved if latest else unresolved[:4])
                    ],
                    "errors": [
                        text(item, 180, latest=latest)
                        for item in (errors if latest else errors[:3])
                    ],
                },
            )

        for item_index, raw in enumerate(value.get("handoff_knowledge", []) or []):
            try:
                item = HandoffKnowledge.model_validate(raw)
            except (ValueError, TypeError):
                continue
            knowledge[(item.topic, item.source)] = (
                report_index,
                item_index,
                {
                    "evidence_check": item.evidence_check,
                    "evidence_sources": [{"tool_call_id": e["tool_call_id"], "tool_name": e["tool_name"], "result_status": e.get("result_status", "unknown"), "step_id": e.get("origin_step"), "worker_id": e.get("origin_worker")} for e in item.resolved_evidence],
                    "unresolved_evidence_refs": list(item.unresolved_evidence_refs),
                    "evidence_errors": dict(item.evidence_errors),
                    "topic": text(item.topic, 160, latest=latest),
                    "source": text(item.source, 240, latest=latest),
                    "usage": text(item.usage, 420, latest=latest),
                    "observed_result": text(item.observed_result, 240, latest=latest),
                    "next_action": text(item.next_action, 240, latest=latest),
                },
            )

    def actionable(values):
        return sorted(
            values,
            key=lambda entry: (
                bool(entry[2].get("next_action")),
                entry[0],
                entry[1],
            ),
            reverse=True,
        )

    candidates = (
        [
            ("review_failures", report_index, item)
            for report_index, item in sorted(
                review_outcomes.values(), key=lambda entry: entry[0], reverse=True
            )
        ]
        + [
            ("worker_notes", report_index, item)
            for report_index, _, item in actionable(knowledge.values())
        ]
    )
    if not candidates:
        return {}

    buckets = (
        "review_failures",
        "worker_notes",
    )
    payload = {key: [] for key in buckets}
    mandatory = [entry for entry in candidates if entry[1] == latest_report_index]
    optional = [entry for entry in candidates if entry[1] != latest_report_index]

    # The newest handoff is never subject to the older-history budget.
    for bucket, _, item in mandatory:
        payload[bucket].append(item)

    selected_optional = 0

    def finalized(candidate_payload, selected_count):
        result = {key: list(candidate_payload[key]) for key in buckets}
        omitted = len(optional) - selected_count
        if omitted:
            result["omitted_entries"] = omitted
            detail = "需要时调用read_execution_history，先填step_ids查看调用，再填references读取结果"
            with_detail = {**result, "details": detail}
            if len(json.dumps(with_detail, ensure_ascii=False, separators=(",", ":"))) <= max_chars:
                result = with_detail
        return result

    for bucket, _, item in optional:
        trial = {key: list(payload[key]) for key in buckets}
        trial[bucket].append(item)
        candidate = finalized(trial, selected_optional + 1)
        if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) <= max_chars:
            payload = trial
            selected_optional += 1

    return finalized(payload, selected_optional)
