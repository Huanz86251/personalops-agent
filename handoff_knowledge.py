"""Portable worker-authored knowledge; preserved without a second summarization."""
from pydantic import BaseModel, ConfigDict, Field, field_validator
from api_handoff import ApiHandoff, ApiHandoffReceipt, validate_apis


def source_results(trace):
    """Collect the bounded sources carried across Worker compaction/review."""
    results = {}
    for key in ("worker_archived_messages", "handoff_source_messages", "messages"):
        for message in trace.get(key, []) or []:
            message = message.model_dump(mode="json") if hasattr(message, "model_dump") else message
            if isinstance(message, dict) and message.get("tool_call_id"):
                results[message["tool_call_id"]] = message
    # Older Code/Web traces may already have a durable resolved-evidence
    # envelope.  It is an equally valid real Tool result source.
    for key in ("worker_submission", "code_worker_submission"):
        record = trace.get(key) or {}
        if hasattr(record, "model_dump"):
            record = record.model_dump(mode="json")
        if not isinstance(record, dict):
            continue
        for item in record.get("resolved_evidence", []) or []:
            if hasattr(item, "model_dump"):
                item = item.model_dump(mode="json")
            if isinstance(item, dict) and item.get("tool_call_id"):
                results.setdefault(item["tool_call_id"], {
                    "tool_call_id": item["tool_call_id"],
                    "name": item.get("tool_name"),
                    "content": item.get("result", ""),
                })
    return results


class HandoffKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apis: list[ApiHandoff] = Field(default_factory=list, description='需传递的API列表。Harness逐项校验，不由Reviewer审批。参数source为null表示尚缺，不代表可执行。')

    @field_validator('apis', mode='before')
    @classmethod
    def parse_each_api(cls, value):
        if not isinstance(value,list): return []
        accepted=[]
        from observability import trace_span,set_span_output
        for index,raw in enumerate(value):
            try: accepted.append(ApiHandoff.model_validate(raw))
            except (ValueError,TypeError):
                with trace_span('Handoff / Invalid API Entry',input_value={'index':index}) as span:
                    set_span_output(span,{'status':'REJECTED','reason':'schema_invalid'})
        return accepted

    topic: str = Field(min_length=1, max_length=300, description="Specific capability or discovery needed by the next task, not a generic progress summary.")
    source: str = Field(min_length=1, max_length=1200, description="Retrievable documentation locator: app/API name, URL, or shared artifact path/section. Do not use a private temporary path, a local short evidence ID, or phrases such as see my previous login result as a locator; the next agent cannot see it.")
    usage: str = Field(min_length=1, max_length=6000, description="Self-contained instructions based on observed docs/results. For an API include exact documented name/signature, purpose, required keyword arguments, authentication and pagination when applicable, plus a minimal call example. Explain placeholders and how to obtain their real values via documented calls; never refer to a previous local variable or copy passwords/tokens. For code/web findings include relevant command/file/URL and how to use it. Do not invent missing details: state what is still unknown.")
    observed_result: str = Field(min_length=1, max_length=3000, description="Observed return type/field names and their meaning, or verified behavior/results; distinguish documentation-only knowledge from an actually executed successful call. Preserve structure/newlines. Never include credentials or unrelated personal records.")
    next_action: str = Field(min_length=1, max_length=2000, description="What the next agent can do with this knowledge, prerequisites it must obtain, and unresolved limits. State whether prior writes already succeeded so they are not repeated.")


def knowledge_field():
    return Field(default_factory=list, max_length=12, description=(
        "Knowledge handoff to a NEW agent that does not automatically receive your conversation, tool outputs, "
        "or private variables; registered execution history may be read on demand. Include all task-relevant discoveries needed to continue without "
        "repeating exploration: exact API usage and return structure, parameter sources, useful "
        "code/web findings, known pitfalls and remaining work. A name alone is insufficient. "
        "Use [] only when there is no reusable knowledge. These are worker-authored claims, "
        "not an independent verification verdict; keep supporting evidence in the existing report."
    ))


def collect_handoff_knowledge(traces):
    """Take accepted submissions only, never model-generated reporter replacements.

    Later corrections of the same topic/source supersede older ones. A local
    evidence alias is deliberately not transported as a globally valid identity.
    """
    selected = {}
    for trace in traces:
        results = source_results(trace)
        for key in ("worker_submission", "code_worker_submission", "general_result"):
            value = trace.get(key) or {}
            if not isinstance(value, dict):
                continue
            value = value.get("submission", value)
            for raw in value.get("handoff_knowledge", []):
                item = HandoffKnowledge.model_validate(raw)
                if item.apis:
                    from observability import trace_span,set_span_output
                    with trace_span('Handoff / Validate API Entries',input_value={'topic':item.topic,'count':len(item.apis)}) as span:
                        apis,errors=validate_apis(item.apis,results)
                        item=item.model_copy(update={'apis':apis})
                        set_span_output(span,{'accepted':len(apis),'errors':errors,'validation':'structure_and_source_only'})
                selected[(item.topic, item.source)] = item
    return list(selected.values())

def collect_api_handoff_receipts(traces):
    """Preserve every structurally valid API handoff with its validation result."""
    selected = {}
    for trace in traces:
        results = source_results(trace)
        for key in ("worker_submission", "code_worker_submission", "general_result", "code_review_report"):
            value = trace.get(key) or {}
            if not isinstance(value, dict):
                continue
            value = value.get("submission", value)
            for raw in value.get("handoff_apis", []) or []:
                try:
                    item = ApiHandoff.model_validate(raw)
                except (ValueError, TypeError):
                    continue
                accepted, errors = validate_apis([item], results)
                receipt = ApiHandoffReceipt(
                    api=item,
                    validation_status="VALIDATED" if accepted else "REJECTED",
                    validation_error="" if accepted else (
                        errors[0].get("detail") or errors[0].get("reason") or "validation failed"
                    ),
                )
                # A later corrected submission supersedes an earlier bad one.
                selected[item.name] = receipt
    return list(selected.values())


def collect_api_handoffs(traces):
    """Flat worker API handoff, independent of reviewer verdict or output."""
    selected={}
    from observability import trace_span,set_span_output
    for trace in traces:
        results = source_results(trace)
        for key in ('worker_submission','code_worker_submission','general_result','code_review_report'):
            value=trace.get(key) or {}
            if not isinstance(value,dict): continue
            value=value.get('submission',value)
            raw=value.get('handoff_apis',[])
            if not raw: continue
            with trace_span('Handoff / Validate API Entries',input_value={'count':len(raw)}) as span:
                items,errors=validate_apis(raw,results)
                for item in items: selected[item.name]=item
                set_span_output(span,{'accepted':len(items),'errors':errors,'validation':'structure_and_source_only'})
    return list(selected.values())


def compact_direct_handoff(reports, max_chars=1800):
    """Keep the newest Worker handoff whole; bound only older handoffs.

    The newest report is the next Worker's primary handoff.  None of its valid
    API receipts, failure details, or worker notes may be omitted by the legacy
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

    validated = {}
    rejected = {}
    knowledge = {}
    review_outcomes = {}

    def text(value, limit, *, latest):
        normalized = str(value or "")
        return normalized if latest else normalized[:limit]

    def compact_api(item, *, latest):
        return {
            "name": item.name,
            "purpose": text(item.purpose, 180, latest=latest),
            "parameters": [
                {
                    "name": parameter.name,
                    "required": parameter.required,
                    "purpose": text(parameter.purpose, 100, latest=latest),
                    "value_source": "已取得" if parameter.source is not None else "尚缺",
                }
                for parameter in item.parameters
            ],
            "call_example": text(item.call_example, 320, latest=latest),
            "next_action": text(item.next_action, 180, latest=latest),
        }

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

        receipts = value.get("handoff_api_receipts", []) or []
        receipt_names = set()
        for item_index, raw in enumerate(receipts):
            try:
                receipt = ApiHandoffReceipt.model_validate(raw)
            except (ValueError, TypeError):
                continue
            item = receipt.api
            receipt_names.add(item.name)
            compact = compact_api(item, latest=latest)
            ranked = (report_index, item_index, compact)
            if receipt.validation_status == "VALIDATED":
                validated[item.name] = ranked
                rejected.pop(item.name, None)
            else:
                compact["validation_error"] = text(
                    receipt.validation_error, 320, latest=latest
                )
                rejected[item.name] = ranked
                validated.pop(item.name, None)

        # Compatibility for reports produced before per-item receipts existed.
        for item_index, raw in enumerate(value.get("handoff_apis", []) or []):
            try:
                item = ApiHandoff.model_validate(raw)
            except (ValueError, TypeError):
                continue
            if item.name in receipt_names:
                continue
            compact = compact_api(item, latest=latest)
            validated[item.name] = (report_index, item_index, compact)
            rejected.pop(item.name, None)

        for item_index, raw in enumerate(value.get("handoff_knowledge", []) or []):
            try:
                item = HandoffKnowledge.model_validate(raw)
            except (ValueError, TypeError):
                continue
            knowledge[(item.topic, item.source)] = (
                report_index,
                item_index,
                {
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
            ("validated_apis", report_index, item)
            for report_index, _, item in actionable(validated.values())
        ]
        + [
            ("worker_notes", report_index, item)
            for report_index, _, item in actionable(knowledge.values())
        ]
        + [
            ("rejected_api_leads", report_index, item)
            for report_index, _, item in actionable(rejected.values())
        ]
    )
    if not candidates:
        return {}

    buckets = (
        "validated_apis",
        "rejected_api_leads",
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
            detail = "需要时调用read_execution_history读取较早执行记录"
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
