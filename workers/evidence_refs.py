"""Short model-facing evidence references; canonical IDs stay in execution state."""
from copy import deepcopy
from contextvars import ContextVar
import json
import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.types import Command
from schema_utils import schema_repair_feedback
from langchain_core.utils.function_calling import convert_to_openai_tool

FIELD_NAMES = {'evidence_tool_call_ids': 'evidence_refs', 'evidence_tool_call_id': 'evidence_ref'}
ACTIVE_EVIDENCE_REFS = ContextVar('active_evidence_refs', default=None)
CONTROL = {'report_general_result', 'complete_web_step', 'publish_worker_progress',
           'submit_code_for_review', 'respond_to_code_review', 'submit_continued_code_for_review',
           'request_code_worker_repair', 'publish_reviewed_candidate', 'submit_code_review',
           'request_additional_capabilities', 'ask_user'}
WEB_RESULT_REF_PATTERN = re.compile(r"^W[A-Z]+[1-9]\d*$")
FINAL_CONCLUSION_MAX_CHARS = 1600
ANSWER_TEXT_MAX_CHARS = 32000
MIGRATED_FINAL_CONCLUSION = (
    "完整可交付正文已由Harness原样迁移至answer_text；执行结果已进入确定性结案。"
)


def migrate_oversized_final_conclusion(args):
    """Repair one lossless Worker handoff shape before Pydantic rejects it."""
    if not isinstance(args, dict):
        return args
    submission = args.get("submission")
    if not isinstance(submission, dict):
        return args
    conclusion = submission.get("final_conclusion")
    if not isinstance(conclusion, str) or len(conclusion.strip()) <= FINAL_CONCLUSION_MAX_CHARS:
        return args
    answer_text = submission.get("answer_text")
    if not str(answer_text or "").strip():
        if len(conclusion.strip()) > ANSWER_TEXT_MAX_CHARS:
            # Lossless migration is impossible; preserve the validation error
            # instead of silently truncating a user deliverable.
            return args
        answer_text = conclusion.strip()
    repaired = deepcopy(args)
    repaired_submission = dict(repaired["submission"])
    repaired_submission["answer_text"] = answer_text
    repaired_submission["final_conclusion"] = MIGRATED_FINAL_CONCLUSION
    repaired["submission"] = repaired_submission
    return repaired


def value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def is_web_result_ref(reference) -> bool:
    return bool(WEB_RESULT_REF_PATTERN.fullmatch(str(reference or "").strip()))


def web_result_evidence(messages) -> dict[str, dict]:
    """Index only concrete result rows emitted by the registered web_search tool."""

    calls = {}
    for message in messages:
        for call in value(message, "tool_calls", []) or []:
            calls[value(call, "id")] = value(call, "name")

    indexed: dict[str, dict] = {}
    conflicts: set[str] = set()

    def visit(item, parent_call_id):
        if isinstance(item, dict):
            source_ref = str(item.get("source_ref") or "").strip()
            href = str(item.get("href") or "").strip()
            if is_web_result_ref(source_ref) and href.startswith(("http://", "https://")):
                record = {
                    "parent_tool_call_id": parent_call_id,
                    "result": {
                        key: item.get(key)
                        for key in ("source_ref", "title", "href", "body")
                        if item.get(key) is not None
                    },
                }
                if source_ref in indexed and indexed[source_ref] != record:
                    conflicts.add(source_ref)
                else:
                    indexed[source_ref] = record
            for nested in item.values():
                visit(nested, parent_call_id)
        elif isinstance(item, list):
            for nested in item:
                visit(nested, parent_call_id)

    for message in messages:
        call_id = value(message, "tool_call_id")
        if not call_id or calls.get(call_id) != "web_search":
            continue
        content = value(message, "content", "")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            continue
        visit(payload, call_id)

    for source_ref in conflicts:
        indexed.pop(source_ref, None)
    return indexed


def model_display_registry(state, refs: dict[str, str]) -> dict[str, str]:
    """Hide successful Web Search envelopes from the evidence namespace.

    Tool-call and ToolMessage IDs must still match in the provider transcript,
    so successful search parents receive an explicitly non-evidence batch ID.
    Concrete rows keep their WA1/WB2-style source_ref in the JSON body. Failed
    searches have no concrete rows and retain their ordinary E reference so a
    Worker can cite the failure itself.
    """

    messages = [
        *state.get("worker_archived_messages", []),
        *state.get("messages", []),
    ]
    web_results = web_result_evidence(messages)
    successful_parents = {
        record["parent_tool_call_id"]
        for record in web_results.values()
    }
    if not successful_parents:
        return refs

    displayed = dict(refs)
    batch_number = 0
    for message in messages:
        for call in value(message, "tool_calls", []) or []:
            call_id = value(call, "id")
            if (
                call_id in successful_parents
                and value(call, "name") == "web_search"
            ):
                batch_number += 1
                displayed[call_id] = f"SEARCH_BATCH_{batch_number}"
    return displayed


def registry(state):
    """Append only; archives and accepted inherited records survive compaction."""
    refs = dict(state.get('worker_evidence_refs', {}))
    def register(call_id):
        if call_id and call_id not in refs:
            refs[call_id] = f'E{len(refs) + 1}'
    messages = [*state.get('worker_archived_messages', []), *state.get('messages', [])]
    for message in messages:
        for call in value(message, 'tool_calls', []) or []:
            register(value(call, 'id'))
        register(value(message, 'tool_call_id'))
    def inherited(obj):
        if hasattr(obj, 'model_dump'): obj = obj.model_dump()
        if isinstance(obj, dict):
            # Only resolved request/result envelopes can introduce inherited evidence.
            if {'tool_call_id', 'tool_name', 'result'} <= obj.keys(): register(obj['tool_call_id'])
            for key, item in obj.items():
                if key not in {'messages', 'worker_archived_messages', 'worker_evidence_refs'}: inherited(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj: inherited(item)
    inherited(state)
    if len(set(refs.values())) != len(refs):
        raise ValueError('Conflicting evidence reference registry')
    return refs


def display(obj, refs):
    """Copy a view, never mutate canonical messages or tool response bodies."""
    if isinstance(obj, dict):
        return {FIELD_NAMES.get(k, k): display(v, refs) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [display(v, refs) for v in obj]
    if isinstance(obj, str):
        if obj in refs: return refs[obj]
        # Embedded JSON is common in runtime context and tool results.
        for raw, short in sorted(refs.items(), key=lambda pair: -len(pair[0])):
            obj = re.sub(r'(?<![\w-])' + re.escape(raw) + r'(?![\w-])', lambda _: short, obj)
        for raw, short in FIELD_NAMES.items(): obj = obj.replace(raw, short)
    return obj


def message_view(message, refs):
    if isinstance(message, dict): return display(message, refs)
    data = display(message.model_dump(), refs)
    # Reconstruct through the existing message class to preserve ToolMessage status,
    # AI tool calls and provider metadata, without changing canonical IDs in state.
    return type(message)(**data)


def schema_view(tool, *, criteria_enabled=False):
    schema = deepcopy(convert_to_openai_tool(tool))
    reviewer = schema.get('function', {}).get('name') == 'submit_code_review'
    def visit(obj):
        if isinstance(obj, dict):
            out = {}
            for key, item in obj.items():
                out[FIELD_NAMES.get(key, key)] = visit(item)
            props = out.get('properties', {})
            if criteria_enabled and 'criterion' in props and 'conclusion' in props:
                props['criterion_id'] = props.pop('criterion')
                props['criterion_id']['description'] = 'Copy C1/C2 from HARNESS_CRITERIA for this Step; do not repeat or rewrite criterion text.'
                out['required'] = ['criterion_id' if k == 'criterion' else k for k in out.get('required', [])]
            for field in ('evidence_refs', 'evidence_ref'):
                was_id_field = any(FIELD_NAMES[k] == field for k in obj.get('properties', {}) if k in FIELD_NAMES)
                if field in props and (was_id_field or reviewer):
                    props[field]['description'] = 'Copy registered evidence references: E1/E2 for whole ordinary tool results, or WA1/WB2 style source_ref for one concrete web result. Never invent references. Failed results only support failure claims.'
            return out
        if isinstance(obj, list): return [visit(v) for v in obj]
        if isinstance(obj, str):
            for raw, short in FIELD_NAMES.items(): obj = obj.replace(raw, short)
        return obj
    converted = visit(schema)
    if hasattr(tool, 'model_copy') and 'function' in converted:
        definition = converted['function']
        return tool.model_copy(update={'args_schema': definition.get('parameters', {}),
                                       'description': definition.get('description', '')})
    return converted


def canonical_arguments(
    args,
    refs,
    *,
    reviewer=False,
    progress=False,
    eligible=None,
    tolerant=False,
):
    inverse = {short: raw for raw, short in refs.items()}
    reverse_fields = {short: raw for raw, short in FIELD_NAMES.items()}
    def restore(item):
        if isinstance(item, dict):
            result = {}
            for key, val in item.items():
                target = key if (reviewer or progress) and key == 'evidence_refs' else reverse_fields.get(key, key)
                if target in result: raise ValueError('Duplicate evidence reference fields')
                if target in FIELD_NAMES or ((reviewer or progress) and target == 'evidence_refs'):
                    values = val if isinstance(val, list) else [val]
                    resolved = []
                    for ref in values:
                        if ref in inverse: resolved.append(inverse[ref])
                        elif ref in refs: resolved.append(ref)  # historical checkpoint compatibility
                        elif progress and isinstance(ref, str) and not re.fullmatch(r'E\d+', ref): resolved.append(ref)
                        elif tolerant:
                            continue
                        else: raise ValueError(f'Unknown evidence reference {ref}; use references shown in tool results.')
                    if eligible is not None and any(raw not in eligible for raw in resolved if not progress or raw in refs):
                        raise ValueError('Evidence must reference a completed business tool result, not a control call or unavailable record.')
                    if isinstance(val, list):
                        result[target] = resolved
                    elif resolved:
                        result[target] = resolved[0]
                    else:
                        # Tolerant Web completion may drop one invalid scalar
                        # reference. Preserve an empty container so the report
                        # can be conservatively downgraded instead of crashing
                        # inside Harness normalization.
                        result[target] = []
                else: result[target] = restore(val)
            return result
        if isinstance(item, list): return [restore(v) for v in item]
        return item
    return restore(args)


def eligible_evidence(state):
    calls, results = {}, set()
    messages = [*state.get('worker_archived_messages', []), *state.get('messages', [])]
    for message in messages:
        for call in value(message, 'tool_calls', []) or []: calls[value(call, 'id')] = value(call, 'name')
        if value(message, 'tool_call_id') and value(message, 'content'):
            results.add(value(message, 'tool_call_id'))
    allowed = {raw for raw, name in calls.items() if raw in results and name not in CONTROL}
    def inherited(obj):
        if hasattr(obj, 'model_dump'): obj = obj.model_dump()
        if isinstance(obj, dict):
            if {'tool_call_id', 'tool_name', 'result'} <= obj.keys() and obj['result'] and obj['tool_name'] not in CONTROL:
                allowed.add(obj['tool_call_id'])
            for k, v in obj.items():
                if k not in {'messages', 'worker_archived_messages'}: inherited(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj: inherited(v)
    inherited(state)
    web_results = web_result_evidence(messages)
    # Once a successful Web Search exposes concrete rows, require claims to cite
    # those rows instead of the whole search envelope.  This keeps a citation
    # such as WA2 bound to one URL/snippet while still allowing the parent call
    # to serve as failure evidence when no concrete result was returned.
    allowed.difference_update(
        record["parent_tool_call_id"] for record in web_results.values()
    )
    allowed.update(web_results)
    return allowed


class EvidenceReferenceMiddleware(AgentMiddleware):
    """Model schemas/views use aliases; tool execution receives canonical IDs."""
    def before_model(self, state, runtime):
        refs = registry(state)
        from reporting.criteria import worker_registry
        criteria = worker_registry(state)
        update = {}
        if refs != state.get('worker_evidence_refs'):
            update['worker_evidence_refs'] = refs
        if criteria != state.get('worker_criterion_refs'):
            update['worker_criterion_refs'] = criteria
        return update or None

    def _request(self, request):
        from reporting.criteria import worker_registry
        refs = registry(request.state)
        allowed_refs = {refs[raw] for raw in eligible_evidence(request.state) if raw in refs}
        display_refs = model_display_registry(request.state, refs)
        messages = [message_view(m, display_refs) for m in request.messages]
        system = message_view(request.system_message, display_refs) if request.system_message else None
        criteria = worker_registry(request.state)
        if criteria and not any('HARNESS_CRITERIA: ' in str(value(m, 'content', '')) for m in messages):
            content = str(value(system, 'content', '')) if system else ''
            system = SystemMessage(content=content + '\nHARNESS_CRITERIA: ' + json.dumps(criteria, ensure_ascii=False))
        for m in messages:
            if isinstance(m, ToolMessage) and m.tool_call_id in allowed_refs:
                # Same short reference as the protocol ID; no duplicate evidence body.
                label = f'[evidence_ref: {m.tool_call_id}]\n'
                if isinstance(m.content, str): m.content = label + m.content
        converted = request.override(messages=messages,
            system_message=system,
            tools=[schema_view(t, criteria_enabled=bool(worker_registry(request.state))) for t in request.tools])
        from workers.tool_grounding import ground_model_request
        return ground_model_request(converted, refs)

    def wrap_model_call(self, request, handler):
        token = ACTIVE_EVIDENCE_REFS.set(registry(request.state))
        try: return handler(self._request(request))
        finally: ACTIVE_EVIDENCE_REFS.reset(token)

    async def awrap_model_call(self, request, handler):
        token = ACTIVE_EVIDENCE_REFS.set(registry(request.state))
        try: return await handler(self._request(request))
        finally: ACTIVE_EVIDENCE_REFS.reset(token)

    def _tool_request(self, request):
        refs = registry(request.state)
        web_refs = web_result_evidence([
            *request.state.get("worker_archived_messages", []),
            *request.state.get("messages", []),
        ])
        refs = {**refs, **{reference: reference for reference in web_refs}}
        from workers.tool_grounding import validate_and_strip_tool_call
        request = validate_and_strip_tool_call(request, refs)
        call = dict(request.tool_call)
        if call.get("name") == "complete_web_step":
            call["args"] = migrate_oversized_final_conclusion(call.get("args", {}))
        # A business tool or progress report may already use an unrelated field
        # named evidence_refs (URLs/files). Only adapt actual evidence-ID contracts.
        original_schema = json.dumps(convert_to_openai_tool(request.tool)) if request.tool else ''
        reviewer = call.get('name') == 'submit_code_review'
        progress = call.get('name') == 'publish_worker_progress'
        if not reviewer and not progress and not any(field in original_schema for field in FIELD_NAMES): return request
        web_completion = call.get('name') == 'complete_web_step'
        call['args'] = canonical_arguments(
            call.get('args', {}),
            refs,
            reviewer=reviewer,
            progress=progress,
            eligible=eligible_evidence(request.state),
            tolerant=web_completion,
        )
        # Resolve API sources without rejecting all entries for one unknown ID.
        def handoff_refs(value, inside=False):
            if isinstance(value,list): return [handoff_refs(v,inside) for v in value]
            if isinstance(value,dict):
                return {k: (next((raw for raw,short in refs.items() if short==v),v)
                            if inside and k=='tool_call_id' and isinstance(v,str)
                            else handoff_refs(v,inside or k in {'handoff_knowledge','handoff_apis'})) for k,v in value.items()}
            return value
        call['args']=handoff_refs(call['args'])
        from reporting.criteria import restore_claims, worker_registry
        if call.get('name') in CONTROL:
            call['args'] = restore_claims(call['args'], worker_registry(request.state))
        return request.override(tool_call=call)

    def wrap_tool_call(self, request, handler):
        try:
            converted = self._tool_request(request)
        except (ValueError, TypeError) as error:
            return self._error(request, error)
        redirected = self._redirect_different_binding(converted)
        if redirected is not None:
            return redirected
        return self._repair_result(converted, handler(converted))

    async def awrap_tool_call(self, request, handler):
        try:
            converted = self._tool_request(request)
        except (ValueError, TypeError) as error:
            return self._error(request, error)
        redirected = self._redirect_different_binding(converted)
        if redirected is not None:
            return redirected
        return self._repair_result(converted, await handler(converted))

    def _redirect_different_binding(self, request):
        from workers.tool_grounding import ACTION_CARD_REREAD_MARKER
        call = dict(request.tool_call)
        args = dict(call.get("args", {}) or {})
        redirect = args.pop(ACTION_CARD_REREAD_MARKER, None)
        if not isinstance(redirect, dict):
            return None
        receipt = str(redirect.get("binding_ref") or "")
        invalid = list(request.state.get("worker_invalid_target_receipts", []))
        if receipt and receipt not in invalid:
            invalid.append(receipt)
        rejected_checks = [
            item for item in redirect.get("rejected_checks", [])
            if isinstance(item, dict)
        ]
        has_contract_conflict = any(
            item.get("assessment") == "CONTRACT_CONFLICT"
            for item in rejected_checks
        )
        details = " ".join(
            f"集合{item.get('set_id')}：判断={item.get('assessment')}；原因={item.get('reason')}"
            for item in rejected_checks
        )
        next_action = (
            "不要自行改写集合合同；请在报告中明确上游范围冲突。"
            if has_contract_conflict
            else "请根据上述原因重新查证并提交新的TARGET_READ。"
        )
        message = ToolMessage(
            content=(
                "ACTION_CARD_REREAD_REQUIRED：读取范围或完整性没有通过，本次写入代码未执行。"
                f"旧回执{receipt}已废弃。{details} {next_action}"
            ),
            tool_call_id=call["id"],
            name=call["name"],
        )
        return Command(update={
            "worker_invalid_target_receipts": invalid,
            "messages": [message],
        })

    def _repair_result(self, request, result):
        if not isinstance(result, ToolMessage):
            return result
        if getattr(result, "status", None) == "error":
            return self._error(request, str(result.content))
        from workers.tool_grounding import target_read_binding_review
        prompt = target_read_binding_review(request, registry(request.state))
        if prompt and isinstance(result.content, str):
            return result.model_copy(update={"content": result.content.rstrip() + "\n" + prompt})
        return result

    def _error(self, request, error):
        refs = registry(request.state)
        web_refs = web_result_evidence([
            *request.state.get("worker_archived_messages", []),
            *request.state.get("messages", []),
        ])
        refs = {**refs, **{reference: reference for reference in web_refs}}
        valid = [refs[raw] for raw in refs if raw in eligible_evidence(request.state)]
        name = request.tool_call['name']
        content = str(error) + '\nAvailable evidence references: ' + ', '.join(valid)
        if name in CONTROL and request.tool is not None:
            from reporting.criteria import worker_registry
            visible_tool = schema_view(request.tool, criteria_enabled=bool(worker_registry(request.state)))
            parameters = convert_to_openai_tool(visible_tool).get("function", {}).get("parameters", {})
            repair_instruction = None
            if name == 'complete_web_step' and 'final_conclusion' in content:
                repair_instruction = (
                    "保留已经得到的业务结果，不要重新调用业务API。"
                    "若final_conclusion超长或包含整篇用户交付文字，"
                    "把完整正文原样移到submission.answer_text，"
                    "将final_conclusion缩成不超过1600字符的执行结论，然后重新提交同一工具。"
                )
            content = schema_repair_feedback(
                schema_name=name,
                schema=parameters,
                error_text=content,
                instruction=repair_instruction,
            )
        message = ToolMessage(
            content=content,
            tool_call_id=request.tool_call['id'],
            name=name,
            status='error',
        )
        terminal = {
            'report_general_result', 'complete_web_step', 'submit_code_for_review',
            'respond_to_code_review', 'submit_continued_code_for_review', 'submit_code_review',
        }
        if name not in terminal:
            return message
        return Command(update={
            'worker_finalize_requested': True,
            'worker_finalize_reason': 'SCHEMA_REPAIR',
            'messages': [message],
        })
