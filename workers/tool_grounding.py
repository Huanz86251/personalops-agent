"""Lightweight, model-facing source references for tool calls.

This module deliberately reuses the existing E references for tool results.  It
adds short references for user/runtime context and validates that references in
a tool call were actually visible.  It does not ask another model to judge the
reason and does not claim that a visible source semantically proves a decision.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
import re

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


SOURCE_FIELD = "source_refs"
MODEL_REF = "MODEL"
CONTROL_TOOLS = {
    "report_general_result", "submit_for_review", "publish_worker_progress",
    "submit_code_for_review", "respond_to_code_review",
    "submit_continued_code_for_review", "request_code_worker_repair",
    "publish_reviewed_candidate", "submit_code_review",
}
APPWORLD_DISCOVER = "appworld_discover"
APPWORLD_EXECUTION = {"appworld_execute", "appworld_verify"}
ACTION_CARD_REREAD_MARKER = "_action_card_reread"


class GroundingMode(str, Enum):
    EXEMPT = "exempt"
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True)
class ToolGroundingPolicy:
    mode: GroundingMode
    reason: str


# One central registry for every currently exposed tool family.  Exact business
# tools are listed so a newly added mutation does not become exempt by accident.
EXEMPT_TOOLS = CONTROL_TOOLS | {
    # Framework/control and terminal-style tools.
    "show_all_toolsets", "execute", "shell",
    # Pure local computation or zero-context inspection.
    "get_current_time", "symbolic_math", "python_syntax_check", "python_static_check",
    "ls", "glob", "grep", "list_directory", "find_files", "grep_files",
    # Harness-owned catalogs: the call itself creates the evidence.
    "search_knowledge", "read_knowledge", "read_execution_history", "read_review_material",
}
OPTIONAL_TOOLS = {
    # Read/search/navigation calls may begin from MODEL or a visible source.
    "read_file", "web_search", "find_github_mirror", "fetch_webpage",
    "attachment_to_text", "ocr_image", "spreadsheet_read",
    "browser_navigate", "browser_snapshot", "browser_find", "browser_tabs",
    "browser_navigate_back", "browser_wait_for", "browser_close",
    "email_connection_status", "email_list_recent", "email_get_snippet",
    "email_read_message", "email_list_attachments", "schedule_list", "schedule_runs",
}
REQUIRED_TOOLS = {
    # Mutations, external actions, captures and AppWorld execution.
    "write_file", "replace_in_file", "edit_file", "delete", "convert_document",
    "spreadsheet_write", "spreadsheet_format", "spreadsheet_chart",
    "schedule_create", "schedule_create_feishu_reminder", "schedule_create_agent_task",
    "schedule_pause", "schedule_resume", "schedule_delete", "windows_notify",
    "capture_desktop_screenshot", "send_local_file_to_feishu",
    "browser_click", "browser_type", "browser_fill_form", "browser_select_option",
    "email_download_attachment", "email_create_draft",
    APPWORLD_DISCOVER, *APPWORLD_EXECUTION,
}


def policy_for(tool_name: str) -> ToolGroundingPolicy:
    if tool_name in EXEMPT_TOOLS:
        return ToolGroundingPolicy(GroundingMode.EXEMPT, "control, terminal, pure computation or harness catalog")
    if tool_name in REQUIRED_TOOLS:
        return ToolGroundingPolicy(GroundingMode.REQUIRED, "mutation or environment-specific execution")
    if tool_name in OPTIONAL_TOOLS:
        return ToolGroundingPolicy(GroundingMode.OPTIONAL, "read, search or navigation")
    # Safe migration default: show the field without blocking an unknown tool.
    return ToolGroundingPolicy(GroundingMode.OPTIONAL, "unclassified tool; audit before enforcing")


def unclassified_tools(names) -> list[str]:
    known = EXEMPT_TOOLS | OPTIONAL_TOOLS | REQUIRED_TOOLS
    return sorted({str(name) for name in names if str(name) not in known})


def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _kwargs(message):
    return _value(message, "additional_kwargs", {}) or {}


def _kind(message) -> str:
    extra = _kwargs(message)
    if extra.get("knowledge_source") or extra.get("knowledge_context"):
        return "R"
    if extra.get("execution_state_catalog"):
        return "S"
    if extra.get("execution_history_catalog"):
        return "H"
    if extra.get("personalops_runtime_event"):
        return "C"
    role = _value(message, "type", "") or _value(message, "role", "")
    return "U" if role in {"human", "user"} else ""


def source_catalog(messages, evidence_refs) -> dict[str, dict]:
    """Build deterministic short references from the model-visible history."""
    catalog: dict[str, dict] = {
        MODEL_REF: {"kind": "model_knowledge", "description": "通用推理或计算；不能单独证明环境API或业务参数"},
        "P1": {"kind": "policy", "description": "当前System与已选择Skill中的规则或示例"},
    }
    counts: dict[str, int] = {}
    for message in messages:
        kind = _kind(message)
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
            catalog[f"{kind}{counts[kind]}"] = {
                "kind": {"U": "user", "R": "rag", "S": "state", "H": "history", "C": "runtime"}[kind],
            }
        call_id = _value(message, "tool_call_id")
        if call_id:
            short = evidence_refs.get(call_id, call_id if re.fullmatch(r"E\d+", str(call_id)) else None)
            if short:
                catalog[short] = {"kind": "tool_result", "canonical_id": call_id}
    return catalog


def _label_messages(messages, catalog):
    counts: dict[str, int] = {}
    result = []
    for message in messages:
        kind = _kind(message)
        if not kind:
            result.append(message)
            continue
        counts[kind] = counts.get(kind, 0) + 1
        ref = f"{kind}{counts[kind]}"
        content = _value(message, "content", "")
        label = f"[source_ref: {ref}]\n"
        if isinstance(content, str) and not content.startswith(label):
            message = message.model_copy(update={"content": label + content})
        result.append(message)
    return result


def schema_with_sources(tool, available_refs):
    name = str(getattr(tool, "name", ""))
    policy = policy_for(name)
    if not name or policy.mode is GroundingMode.EXEMPT:
        return tool
    converted = deepcopy(convert_to_openai_tool(tool))
    definition = converted.get("function", {})
    parameters = definition.get("parameters", {})
    properties = parameters.setdefault("properties", {})
    if SOURCE_FIELD not in properties:
        properties[SOURCE_FIELD] = {
            # Keep the execution schema byte-for-byte stable across model
            # rounds.  The visible message labels are the live catalog and
            # validate_and_strip_tool_call checks membership at runtime.
            "type": "array", "items": {
                "type": "string",
                "pattern": r"^(?:MODEL|P1|[URSHCE][1-9][0-9]*)$",
            },
            "minItems": 1, "maxItems": 4, "uniqueItems": True,
            "description": (
                "复制当前消息中实际可见的短来源号（如U1、R1、E2）。"
                "可用MODEL表示通用知识；"
                "它不能单独证明环境中的具体接口、字段、路径或业务参数。"
            ),
        }
    required = list(parameters.get("required", []))
    if policy.mode is GroundingMode.REQUIRED and SOURCE_FIELD not in required:
        required.insert(0, SOURCE_FIELD)
    parameters["required"] = required
    if hasattr(tool, "model_copy"):
        return tool.model_copy(update={"args_schema": parameters,
                                       "description": definition.get("description", "")})
    return converted


def ground_model_request(request, evidence_refs):
    messages = list(request.messages)
    catalog = source_catalog(messages, evidence_refs)
    labeled = _label_messages(messages, catalog)
    system = request.system_message
    if system:
        content = _value(system, "content", "")
        guide = (
            "\n\nTOOL_SOURCE_REFS: 调用业务工具时填写source_refs，只复制当前可见编号；"
            "MODEL仅表示通用知识。引用用于调用前自查和Harness审计，不要求展开长篇理由。"
        )
        if isinstance(content, str) and "TOOL_SOURCE_REFS:" not in content:
            system = system.model_copy(update={
                "content": "[source_ref: P1]\n" + content.rstrip() + guide})
    refs = tuple(catalog)
    return request.override(
        messages=labeled,
        system_message=system,
        tools=[schema_with_sources(tool, refs) for tool in request.tools],
    )


def _tool_result_index(state, evidence_refs):
    """Map short refs to successful calls and materialize frozen read receipts."""
    calls = {}
    results = {}
    messages = [*state.get("worker_archived_messages", []), *state.get("messages", [])]
    for message in messages:
        for call in _value(message, "tool_calls", []) or []:
            calls[_value(call, "id")] = {
                "tool": _value(call, "name"),
                "args": dict(_value(call, "args", {}) or {}),
            }
        call_id = _value(message, "tool_call_id")
        if call_id:
            results[call_id] = _value(message, "status", None) != "error"

    indexed = {}
    for raw, details in calls.items():
        if raw not in evidence_refs or raw not in results:
            continue
        ref = evidence_refs[raw]
        args = deepcopy(details["args"])
        phase = args.get("action_phase")
        receipt_ref = args.get("binding_ref")
        if phase in {"TARGET_WRITE", "TARGET_VERIFY"} and receipt_ref:
            receipt = indexed.get(receipt_ref, {})
            receipt_args = receipt.get("args", {}) if isinstance(receipt, dict) else {}
            if (
                receipt.get("succeeded")
                and receipt.get("tool") in APPWORLD_EXECUTION
                and receipt_args.get("action_phase") == "TARGET_READ"
            ):
                args["set_bindings"] = [
                    {**deepcopy(item), "source_ref": receipt_ref}
                    for item in receipt_args.get("set_bindings", [])
                    if isinstance(item, dict)
                ]
        indexed[ref] = {
            "tool": details["tool"],
            "args": args,
            "succeeded": results[raw],
        }
    return indexed


def _api_is_called(code: str, api_name: str) -> bool:
    """Use an exact dotted-call boundary, not substring matching."""
    import re

    return bool(re.search(
        rf"(?<![A-Za-z0-9_])apis\.{re.escape(api_name)}\s*\(",
        code,
    ))


def _binding_signature(bindings, *, include_source=False):
    return tuple(sorted(
        (
            str(item.get("set_id") or ""),
            str(item.get("read_api") or ""),
            *(
                (str(item.get("source_ref") or ""),)
                if include_source else ()
            ),
        )
        for item in bindings if isinstance(item, dict)
    ))


def _action_card_error(code, problem, correction, example=None):
    lines = [
        f"ACTION_CARD_REJECTED[{code}]",
        f"问题：{problem}",
        f"下一次必须：{correction}",
    ]
    if example:
        lines.append(f"正确结构示例：{example}")
    lines.append("请修改Action Card后重新提交；不要原样重复本次调用。")
    return ValueError("\n".join(lines))


def _successful_phase_records(prior, phase):
    return [
        (ref, details)
        for ref, details in prior.items()
        if details.get("succeeded")
        and details.get("tool") in APPWORLD_EXECUTION
        and details.get("args", {}).get("action_phase") == phase
    ]


def _validate_appworld_finalize_chain(prior, refs, effect_mode="MUTATION"):
    """Require the completed phase chain appropriate for the external effect."""

    reads = _successful_phase_records(prior, "TARGET_READ")
    writes = _successful_phase_records(prior, "TARGET_WRITE")
    verifies = _successful_phase_records(prior, "TARGET_VERIFY")
    if effect_mode == "READ_ONLY":
        if not reads:
            raise _action_card_error(
                "FINALIZE_MISSING_READ",
                "READ_ONLY FINALIZE requires one prior successful TARGET_READ.",
                "先完成目标读取，再单独提交FINALIZE。",
            )
        referenced_reads = [details for ref, details in reads if ref in refs]
        if not referenced_reads:
            available = [ref for ref, _ in reads]
            raise _action_card_error(
                "FINALIZE_READ_REF",
                "READ_ONLY FINALIZE must cite a successful TARGET_READ result.",
                f"source_refs中引用成功读取编号；当前可用={available}。",
            )
        return
    missing = [
        label for label, records in (
            ("TARGET_READ", reads),
            ("TARGET_WRITE", writes),
            ("TARGET_VERIFY", verifies),
        )
        if not records
    ]
    if missing:
        raise _action_card_error(
            "FINALIZE_MISSING_PHASES",
            "FINALIZE requires prior successful target phases: " + ", ".join(missing),
            "先依次完成缺失阶段，再单独提交FINALIZE。",
            '{"action_phase":"FINALIZE","set_bindings":[],"source_refs":["<成功TARGET_VERIFY的E编号>"]}',
        )

    write_lineages = {
        _binding_signature(
            details.get("args", {}).get("set_bindings", []),
            include_source=True,
        )
        for _, details in writes
    }
    referenced_verifies = [
        details for ref, details in verifies if ref in refs
    ]
    if not referenced_verifies:
        available = [ref for ref, _ in verifies]
        raise _action_card_error(
            "FINALIZE_VERIFY_REF",
            "FINALIZE must cite the E reference of a successful TARGET_VERIFY result.",
            f"source_refs中引用成功回读编号；当前可用={available}。",
        )
    if not any(
        _binding_signature(
            details.get("args", {}).get("set_bindings", []),
            include_source=True,
        ) in write_lineages
        for details in referenced_verifies
    ):
        raise _action_card_error(
            "FINALIZE_LINEAGE_MISMATCH",
            "The cited TARGET_VERIFY must use the same dynamic TARGET_READ receipt "
            "and set-to-API bindings as a successful TARGET_WRITE.",
            "回读与写入必须引用同一次TARGET_READ冻结回执，并保留完全相同的set_id→read_api映射。",
        )


def _frozen_bindings_from_receipt(prior, receipt_ref):
    receipt = prior.get(receipt_ref, {}) if isinstance(receipt_ref, str) else {}
    receipt_args = receipt.get("args", {}) if isinstance(receipt, dict) else {}
    if not (
        receipt.get("succeeded")
        and receipt.get("tool") in APPWORLD_EXECUTION
        and receipt_args.get("action_phase") == "TARGET_READ"
    ):
        return None
    bindings = receipt_args.get("set_bindings", [])
    if not isinstance(bindings, list) or not bindings:
        return None
    return [
        {**deepcopy(item), "source_ref": receipt_ref}
        for item in bindings if isinstance(item, dict)
    ]


def _validate_appworld_target_bindings(request, args, refs, prior):
    """Bind Scheduler set operands to observed AppWorld reads.

    This validates coverage and provenance. It deliberately does not claim to
    understand the business meaning of an API name; the model must state that
    comparison in read_api_reason before the call can execute.
    """

    selection = request.state.get("worker_target_selection")
    if not isinstance(selection, dict):
        return

    phase = args.get("action_phase")
    bindings = args.get("set_bindings")
    binding_ref = args.get("binding_ref")
    binding_checks = args.get("binding_checks", [])
    effect_mode = str(selection.get("effect_mode") or "MUTATION")
    current_code = str(args.get("code") or "")
    completes_task = _api_is_called(current_code, "supervisor.complete_task")
    allowed = {
        "PREREQUISITE", "TARGET_READ", "TARGET_WRITE",
        "TARGET_VERIFY", "FINALIZE", "OTHER",
    }
    if phase not in allowed:
        raise _action_card_error(
            "PHASE_REQUIRED",
            f"A Step with target_selection requires an explicit valid action_phase; received={phase!r}.",
            "从PREREQUISITE、TARGET_READ、TARGET_WRITE、TARGET_VERIFY、FINALIZE中选择一个。",
        )
    if not isinstance(binding_checks, list):
        raise _action_card_error(
            "BINDING_CHECKS_TYPE",
            f"binding_checks must be a list; received={type(binding_checks).__name__}.",
            "即使本阶段不复核，也必须显式填写binding_checks: []。",
        )
    if not isinstance(bindings, list):
        raise _action_card_error(
            "BINDINGS_TYPE",
            f"set_bindings must be a list; received={type(bindings).__name__}.",
            "即使本阶段无集合绑定，也必须显式填写set_bindings: []。",
        )

    if phase != "FINALIZE" and completes_task:
        raise _action_card_error(
            "FINALIZE_MUST_BE_SEPARATE",
            "complete_task is allowed only in a separate FINALIZE call after a successful TARGET_VERIFY result.",
            "删除本轮complete_task；先完成并引用TARGET_VERIFY，下一轮再用FINALIZE单独调用。",
        )
    if phase == "FINALIZE":
        if binding_ref is not None:
            raise _action_card_error(
                "FINALIZE_RECEIPT",
                "FINALIZE does not accept binding_ref.",
                "填写binding_ref: null，并在source_refs引用成功TARGET_VERIFY结果。",
            )
        if bindings:
            raise _action_card_error(
                "FINALIZE_BINDINGS",
                f"FINALIZE must not claim target-set bindings; received {len(bindings)} item(s).",
                "填写set_bindings: []。",
            )
        if not completes_task:
            raise _action_card_error(
                "FINALIZE_CALL_MISSING",
                "FINALIZE must contain the documented supervisor.complete_task call.",
                "用已确认的真实签名单独调用complete_task，并引用成功TARGET_VERIFY的E编号。",
            )
        _validate_appworld_finalize_chain(prior, refs, effect_mode)
        return
    if effect_mode == "READ_ONLY" and phase in {"TARGET_WRITE", "TARGET_VERIFY"}:
        raise _action_card_error(
            "READ_ONLY_MUTATION_FORBIDDEN",
            f"{phase} is not part of a READ_ONLY target_selection.",
            "只读任务完成TARGET_READ后直接FINALIZE；不要执行写入或写后验收。",
        )
    if phase == "PREREQUISITE":
        if binding_ref is not None:
            raise _action_card_error(
                "PREREQUISITE_RECEIPT",
                "PREREQUISITE does not accept binding_ref.",
                "准备阶段填写binding_ref: null。",
            )
        if bindings:
            raise _action_card_error(
                "PREREQUISITE_BINDINGS",
                f"PREREQUISITE must not claim target-set bindings; received {len(bindings)} item(s).",
                "登录、凭据或其他准备调用填写set_bindings: []。",
            )
        return
    if phase == "OTHER":
        raise _action_card_error(
            "PHASE_OTHER_FORBIDDEN",
            "A Step with target_selection cannot use OTHER.",
            "按本轮真实目的改选PREREQUISITE、TARGET_READ、TARGET_WRITE、TARGET_VERIFY或FINALIZE。",
        )

    if phase != "TARGET_WRITE" and binding_checks:
        raise _action_card_error(
            "BINDING_CHECKS_PHASE",
            "binding_checks are only accepted on MUTATION TARGET_WRITE.",
            "本阶段填写binding_checks: []。",
        )

    operands = selection.get("operands")
    if not isinstance(operands, list) or not operands:
        raise ValueError("worker_target_selection has no valid operands.")

    if phase == "TARGET_READ":
        if binding_ref is not None:
            raise _action_card_error(
                "TARGET_READ_RECEIPT",
                "TARGET_READ creates a receipt and cannot consume binding_ref.",
                "填写binding_ref: null，并完整提交本次set_bindings。",
            )
    else:
        if not isinstance(binding_ref, str) or not binding_ref:
            raise _action_card_error(
                "TARGET_RECEIPT_REQUIRED",
                f"{phase} must cite one frozen TARGET_READ receipt.",
                "把成功TARGET_READ返回的E编号同时填入binding_ref和source_refs；set_bindings填[]。",
                '{"binding_ref":"E5","set_bindings":[],"source_refs":["E5"]}',
            )
        invalid_receipts = set(request.state.get("worker_invalid_target_receipts", []))
        if binding_ref in invalid_receipts:
            raise _action_card_error(
                "TARGET_RECEIPT_SUPERSEDED",
                f"binding_ref={binding_ref!r} was rejected by a prior non-complete read assessment.",
                "重新执行TARGET_READ并使用新返回的E编号；不要复用已废弃回执。",
            )
        if binding_ref not in refs:
            raise _action_card_error(
                "TARGET_RECEIPT_NOT_CITED",
                f"binding_ref={binding_ref!r} must also appear in source_refs={refs}.",
                f"把{binding_ref!r}加入source_refs。",
            )
        if bindings:
            raise _action_card_error(
                "TARGET_RECEIPT_DUPLICATED",
                f"{phase} must not recopy set_bindings when binding_ref is present.",
                "删除重复绑定并填写set_bindings: []；Harness会从回执恢复。",
            )
        frozen = _frozen_bindings_from_receipt(prior, binding_ref)
        if frozen is None:
            available = [ref for ref, _ in _successful_phase_records(prior, "TARGET_READ")]
            raise _action_card_error(
                "TARGET_RECEIPT_INVALID",
                f"binding_ref={binding_ref!r} is not a successful TARGET_READ receipt.",
                f"改用成功TARGET_READ编号；当前可用={available}。",
            )
        args["set_bindings"] = frozen
        bindings = frozen

    if phase == "TARGET_WRITE":
        check_ids = [
            item.get("set_id") for item in binding_checks if isinstance(item, dict)
        ]
        if (
            len(check_ids) != len(binding_checks)
            or len(check_ids) != len(set(check_ids))
            or set(check_ids) != set(operands)
        ):
            raise _action_card_error(
                "BINDING_CHECKS_COVERAGE",
                f"TARGET_WRITE binding_checks must cover each operand once; received={check_ids}, expected={operands}.",
                "按上一轮动态复核提示逐集合填写；每项字段顺序为reason、assessment、set_id。",
            )
        valid_assessments = {
            "COMPLETE_MATCH", "INCOMPLETE_RESULT", "WRONG_READ", "CONTRACT_CONFLICT"
        }
        invalid_assessments = [
            item for item in binding_checks
            if item.get("assessment") not in valid_assessments
            or not str(item.get("reason") or "").strip()
        ]
        if invalid_assessments:
            raise _action_card_error(
                "BINDING_CHECKS_FIELDS",
                "Each binding check requires reason followed by a valid assessment.",
                "先根据真实返回比较范围与完整性并填写reason，再填写assessment。",
            )
        rejected_checks = [
            {
                "set_id": str(item.get("set_id")),
                "assessment": str(item.get("assessment")),
                "reason": str(item.get("reason") or "").strip(),
            }
            for item in binding_checks
            if item.get("assessment") != "COMPLETE_MATCH"
        ]
        if rejected_checks:
            args[ACTION_CARD_REREAD_MARKER] = {
                "binding_ref": binding_ref,
                "rejected_checks": rejected_checks,
            }

    set_ids = [item.get("set_id") for item in bindings if isinstance(item, dict)]
    if len(set_ids) != len(bindings) or len(set_ids) != len(set(set_ids)):
        raise _action_card_error(
            "BINDINGS_STRUCTURE",
            f"set_bindings must contain unique structured set_id entries; received={set_ids}.",
            f"每个集合只填一次；必须覆盖operands={operands}。",
        )
    if set(set_ids) != set(operands):
        missing = sorted(set(operands) - set(set_ids))
        extra = sorted(set(set_ids) - set(operands))
        raise _action_card_error(
            "BINDINGS_COVERAGE",
            "set_bindings must cover every target_selection operand exactly once; "
            f"missing={missing}, extra={extra}.",
            f"提交且只提交这些集合：{operands}；每项包含set_id/read_api_reason/read_api/source_ref/requested_scope/coverage_reason/completion_condition。",
        )

    for binding in bindings:
        api_name = str(binding.get("read_api") or "").strip()
        source_ref = str(binding.get("source_ref") or "").strip()
        read_api_reason = str(binding.get("read_api_reason") or "").strip()
        requested_scope = str(binding.get("requested_scope") or "").strip()
        coverage_reason = str(binding.get("coverage_reason") or "").strip()
        completion_condition = str(binding.get("completion_condition") or "").strip()
        if source_ref not in refs:
            raise _action_card_error(
                "BINDING_SOURCE_NOT_CITED",
                f"set binding {binding.get('set_id')} source_ref={source_ref!r} must also appear in source_refs={refs}.",
                f"把{source_ref!r}加入顶层source_refs，或改用其中已有且真正支持该绑定的编号。",
            )
        if not all((api_name, read_api_reason, requested_scope, coverage_reason, completion_condition)):
            raise _action_card_error(
                "BINDING_FIELDS",
                "Each set binding requires read_api_reason, read_api, requested_scope, "
                f"coverage_reason and completion_condition; set_id={binding.get('set_id')!r}.",
                "按扁平字段补齐真实读取接口、请求范围、完整性理由和停止条件；不要提交coverage_plan对象。",
            )

        if phase == "TARGET_READ":
            if not _api_is_called(current_code, api_name):
                raise _action_card_error(
                    "TARGET_READ_API_NOT_CALLED",
                    f"TARGET_READ declares {api_name}, but the current code does not call it.",
                    f"要么在代码中真实调用apis.{api_name}(...), 要么把read_api改为代码实际用于产生该集合的已确认接口。",
                )
            continue

        source = prior.get(source_ref, {})
        source_args = source.get("args", {}) if isinstance(source, dict) else {}
        source_code = str(source_args.get("code") or "")
        source_bindings = {
            str(item.get("set_id") or ""): str(item.get("read_api") or "")
            for item in source_args.get("set_bindings", [])
            if isinstance(item, dict)
        }
        if not (
            source.get("succeeded")
            and source.get("tool") in APPWORLD_EXECUTION
            and source_args.get("action_phase") == "TARGET_READ"
            and source_bindings.get(str(binding.get("set_id") or "")) == api_name
            and _api_is_called(source_code, api_name)
        ):
            reads = _successful_phase_records(prior, "TARGET_READ")
            expected = {
                ref: {
                    str(item.get("set_id") or ""): str(item.get("read_api") or "")
                    for item in details.get("args", {}).get("set_bindings", [])
                    if isinstance(item, dict)
                }
                for ref, details in reads
            }
            raise _action_card_error(
                "TARGET_LINEAGE",
                f"{phase} binding for {binding.get('set_id')} must cite a prior successful "
                f"TARGET_READ result that actually bound and called {api_name}.",
                "TARGET_WRITE/TARGET_VERIFY中的read_api不是本轮写入或回读接口；"
                f"必须复制冻结TARGET_READ的set_id→read_api，并把source_ref设为该回执。可用映射={expected}。",
            )


def validate_and_strip_tool_call(request, evidence_refs):
    call = dict(request.tool_call)
    name = call.get("name", "")
    policy = policy_for(name)
    if policy.mode is GroundingMode.EXEMPT:
        return request
    args = dict(call.get("args", {}) or {})
    if name in APPWORLD_EXECUTION:
        # Grounding runs before LangChain/Pydantic invokes the concrete tool,
        # so normalize common provider sentinels here as well as in the tool
        # schema.  Otherwise a literal "null" is mistaken for an evidence ID.
        binding_ref = args.get("binding_ref")
        if binding_ref is False or (
            isinstance(binding_ref, str)
            and binding_ref.strip().lower() in {
                "", "null", "none", "nil", "false", "n/a",
            }
        ):
            args["binding_ref"] = None
        for field in ("binding_checks", "set_bindings"):
            value = args.get(field)
            if value is None or value is False or (
                isinstance(value, str)
                and value.strip().lower() in {
                    "", "null", "none", "nil", "false", "n/a",
                }
            ):
                args[field] = []
    refs = args.get(SOURCE_FIELD)
    if refs is None and policy.mode is GroundingMode.OPTIONAL:
        # Compatibility for deterministic fixtures and old checkpoints.  New
        # model schemas still require the field, while an omitted low-risk
        # declaration is recorded as ungrounded model knowledge.
        refs = [MODEL_REF]
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs):
        raise ValueError("source_refs is required; copy 1-4 visible source references or use MODEL.")
    if len(refs) > 4 or len(set(refs)) != len(refs):
        raise ValueError("source_refs must contain 1-4 unique references.")
    catalog = source_catalog([*request.state.get("worker_archived_messages", []),
                              *request.state.get("messages", [])], evidence_refs)
    unknown = [ref for ref in refs if ref not in catalog]
    if unknown:
        raise ValueError("Unknown source_refs: " + ", ".join(unknown))
    if name in APPWORLD_EXECUTION:
        prior = _tool_result_index(request.state, evidence_refs)
        grounded_by_result = any(
            prior.get(ref, {}).get("succeeded") and
            prior.get(ref, {}).get("tool") in {APPWORLD_DISCOVER, *APPWORLD_EXECUTION}
            for ref in refs
        )
        grounded_by_context = any(ref != MODEL_REF and ref in catalog for ref in refs)
        if not (grounded_by_result or grounded_by_context):
            raise ValueError(
                "AppWorld execution cannot rely on MODEL alone. Cite a visible user/RAG/state/history source, "
                "or use appworld_discover to confirm the exact API/signature first."
            )
        _validate_appworld_target_bindings(request, args, refs, prior)
    # Model-only schema augmentation must not leak an unexpected keyword into
    # ordinary tools.  Tools that natively declare source_refs receive it for
    # their own audit record.
    original = convert_to_openai_tool(request.tool) if request.tool else {}
    native = SOURCE_FIELD in original.get("function", {}).get("parameters", {}).get("properties", {})
    if not native:
        args.pop(SOURCE_FIELD, None)
    call["args"] = args
    return request.override(tool_call=call)


def _visible_source_texts(state, evidence_refs):
    """Return model-visible source bodies keyed by their short reference."""
    texts = {}
    counts = {}
    messages = [*state.get("worker_archived_messages", []), *state.get("messages", [])]
    for message in messages:
        kind = _kind(message)
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
            content = _value(message, "content", "")
            if isinstance(content, str):
                texts[f"{kind}{counts[kind]}"] = content
        call_id = _value(message, "tool_call_id")
        if call_id:
            short = evidence_refs.get(
                call_id,
                call_id if re.fullmatch(r"E\d+", str(call_id)) else None,
            )
            content = _value(message, "content", "")
            if short and isinstance(content, str):
                texts[short] = content
    return texts


def _description_excerpt(text, api_name, limit=420):
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return "引用来源没有可显示的描述；不能据此确认范围相同。"
    method = str(api_name or "").split(".")[-1]
    positions = [pos for pos in (compact.find(str(api_name)), compact.find(method)) if pos >= 0]
    if not positions:
        return "引用来源中没有定位到该接口的描述；写入前应判为WRONG_READ并重新查证。"
    pos = min(positions)
    start = max(0, pos - 90)
    end = min(len(compact), pos + limit)
    excerpt = compact[start:end]
    return ("…" if start else "") + excerpt + ("…" if end < len(compact) else "")


def target_read_binding_review(request, evidence_refs):
    """Build a small next-turn semantic check from exact cited source text."""
    args = dict(_value(request, "tool_call", {}).get("args", {}) or {})
    if args.get("action_phase") != "TARGET_READ":
        return ""
    selection = request.state.get("worker_target_selection")
    if not isinstance(selection, dict) or selection.get("effect_mode", "MUTATION") != "MUTATION":
        return ""
    definitions = {
        str(item.get("set_id") or ""): str(item.get("definition") or "")
        for item in selection.get("sets", [])
        if isinstance(item, dict)
    }
    texts = _visible_source_texts(request.state, evidence_refs)
    system_content = _value(_value(request, "system_message"), "content", "")
    if isinstance(system_content, str) and system_content.strip():
        texts["P1"] = system_content
    lines = [
        "",
        "TARGET_BINDING_REVIEW_REQUIRED：下一次准备TARGET_WRITE时，先复核这次读取是否真的对应目标集合。",
        "每项先写reason，再写assessment，最后写set_id。",
        "只有全部COMPLETE_MATCH才可写入；INCOMPLETE_RESULT或WRONG_READ会废弃旧回执并重新TARGET_READ；CONTRACT_CONFLICT会阻止写入并报告范围冲突。",
    ]
    for binding in args.get("set_bindings", []):
        if not isinstance(binding, dict):
            continue
        set_id = str(binding.get("set_id") or "")
        source_ref = str(binding.get("source_ref") or "")
        lines.extend([
            f"- 集合{set_id}定义：{definitions.get(set_id, '')}",
            f"  已引用来源中的真实描述：{_description_excerpt(texts.get(source_ref, ''), binding.get('read_api'))}",
            f"  读取前声明范围：{str(binding.get('requested_scope') or '')}",
            f"  读取前完整条件：{str(binding.get('completion_condition') or '')}",
            f"  读取前完整理由：{str(binding.get('coverage_reason') or '')}",
            f"  上一轮接口适用理由：{str(binding.get('read_api_reason') or '')}",
        ])
    return "\n".join(lines)


def trace_source_refs(request) -> list[str]:
    args = _value(request, "tool_call", {}).get("args", {})
    refs = args.get(SOURCE_FIELD, []) if isinstance(args, dict) else []
    return list(refs) if isinstance(refs, list) else []


__all__ = ["ground_model_request", "validate_and_strip_tool_call", "schema_with_sources",
           "source_catalog", "policy_for", "unclassified_tools", "GroundingMode",
           "target_read_binding_review", "ACTION_CARD_REREAD_MARKER",
           "EXEMPT_TOOLS", "OPTIONAL_TOOLS", "REQUIRED_TOOLS", "SOURCE_FIELD", "MODEL_REF"]
