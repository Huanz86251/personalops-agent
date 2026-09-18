from dataclasses import dataclass, replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from evals.appworld.adapter import make_execute_tool
from workers.tool_grounding import (
    GroundingMode, policy_for, schema_with_sources, source_catalog,
    unclassified_tools, validate_and_strip_tool_call, target_read_binding_review,
    ACTION_CARD_REREAD_MARKER,
)


@tool
def lookup(value: str) -> str:
    """Look up a fixture value."""
    return value


@dataclass
class Request:
    tool_call: dict
    tool: object
    state: dict

    def override(self, **changes):
        return replace(self, **changes)


class World:
    def execute(self, code):
        return "ok"


def test_generic_tool_schema_requires_short_sources_but_runtime_strips_model_field():
    viewed = schema_with_sources(lookup, ["MODEL", "U1", "E1"])
    schema = viewed.args_schema
    assert "source_refs" not in schema.get("required", [])
    assert "enum" not in schema["properties"]["source_refs"]["items"]
    assert schema["properties"]["source_refs"]["items"]["pattern"]
    request = Request({"name": "lookup", "id": "call", "args": {
        "source_refs": ["U1"], "value": "x"}}, lookup,
        {"messages": [HumanMessage(content="lookup x")]})
    grounded = validate_and_strip_tool_call(request, {})
    assert grounded.tool_call["args"] == {"value": "x"}


def test_catalog_distinguishes_user_rag_state_and_real_tool_results():
    messages = [
        HumanMessage(content="task"),
        HumanMessage(content="candidate", additional_kwargs={
            "personalops_runtime_event": True, "knowledge_source": True}),
        HumanMessage(content="state", additional_kwargs={"execution_state_catalog": True}),
        AIMessage(content="", tool_calls=[{"name": "lookup", "id": "raw", "args": {}}]),
        ToolMessage(content="value", tool_call_id="raw", name="lookup"),
    ]
    catalog = source_catalog(messages, {"raw": "E1"})
    assert set(catalog) == {"MODEL", "P1", "U1", "R1", "S1", "E1"}


def test_appworld_execute_accepts_visible_user_or_rag_ref_but_rejects_model_only():
    execute = make_execute_tool(World())
    state = {"messages": [HumanMessage(content="task")]}
    with pytest.raises(ValueError, match="cannot rely on MODEL alone"):
        validate_and_strip_tool_call(Request({"name": "appworld_execute", "id": "x",
            "args": {"source_refs": ["MODEL"], "reason": "guess", "code": "print(1)"}},
            execute, state), {})
    grounded = validate_and_strip_tool_call(Request({"name": "appworld_execute", "id": "x",
        "args": {"source_refs": ["U1"], "reason": "user source", "code": "print(1)"}},
        execute, state), {})
    assert grounded.tool_call["args"]["source_refs"] == ["U1"]


def test_appworld_execute_accepts_successful_discover_result_and_rejects_unknown_ref():
    execute = make_execute_tool(World())
    state = {"messages": [
        AIMessage(content="", tool_calls=[{"name": "appworld_discover", "id": "raw", "args": {}}]),
        ToolMessage(content="signature", tool_call_id="raw", name="appworld_discover"),
    ]}
    grounded = validate_and_strip_tool_call(Request({"name": "appworld_execute", "id": "x",
        "args": {"source_refs": ["E1"], "reason": "signature", "code": "print(1)"}},
        execute, state), {"raw": "E1"})
    assert grounded.tool_call["args"]["source_refs"] == ["E1"]
    with pytest.raises(ValueError, match="Unknown source_refs"):
        validate_and_strip_tool_call(Request({"name": "appworld_execute", "id": "x",
            "args": {"source_refs": ["E9"], "reason": "bad", "code": "print(1)"}},
            execute, state), {"raw": "E1"})


def test_current_tool_inventory_is_explicitly_classified_and_terminal_is_exempt():
    from tools import ALL_TOOLS
    current = {tool.name for tool in ALL_TOOLS} | {
        "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute",
        "search_knowledge", "read_knowledge", "read_execution_history", "read_review_material",
        "appworld_discover", "appworld_execute", "appworld_verify",
        "browser_click", "browser_close", "browser_fill_form", "browser_find",
        "browser_navigate", "browser_navigate_back", "browser_select_option",
        "browser_snapshot", "browser_tabs", "browser_type", "browser_wait_for",
        "email_connection_status", "email_list_recent", "email_get_snippet",
        "email_read_message", "email_list_attachments", "email_download_attachment",
        "email_create_draft",
    }
    assert unclassified_tools(current) == []
    assert policy_for("execute").mode is GroundingMode.EXEMPT
    assert policy_for("web_search").mode is GroundingMode.OPTIONAL
    assert policy_for("write_file").mode is GroundingMode.REQUIRED


def test_exempt_tool_schema_has_no_source_field():
    terminal = lookup.model_copy(update={"name": "execute"})
    viewed = schema_with_sources(terminal, ["MODEL", "U1"])
    schema = viewed.args_schema.model_json_schema()
    assert "source_refs" not in schema.get("properties", {})


def test_required_generic_tool_rejects_missing_source_and_strips_valid_source():
    writer = lookup.model_copy(update={"name": "write_file"})
    state = {"messages": [HumanMessage(content="write the requested value")]}
    with pytest.raises(ValueError, match="source_refs is required"):
        validate_and_strip_tool_call(Request(
            {"name": "write_file", "id": "call", "args": {"value": "x"}},
            writer,
            state,
        ), {})
    grounded = validate_and_strip_tool_call(Request(
        {"name": "write_file", "id": "call", "args": {
            "source_refs": ["U1"], "value": "x"}},
        writer,
        state,
    ), {})
    assert grounded.tool_call["args"] == {"value": "x"}


def test_generic_source_schema_is_stable_when_visible_refs_grow():
    first = schema_with_sources(lookup, ["MODEL", "U1"])
    later = schema_with_sources(lookup, ["MODEL", "U1", "E1", "E2", "E3"])
    assert first.args_schema == later.args_schema


def _target_selection_state(messages=None, effect_mode="MUTATION"):
    return {
        "messages": messages or [HumanMessage(content="处理文件夹内且已标星的文档")],
        "worker_target_selection": {
            "target_entity": "document",
            "effect_mode": effect_mode,
            "sets": [
                {"set_id": "A", "definition": "文件夹内的文档",
                 "condition_owner": "folder", "result_entity": "document"},
                {"set_id": "B", "definition": "已标星的文档",
                 "condition_owner": "document", "result_entity": "document"},
            ],
            "operation": "INTERSECTION",
            "operands": ["A", "B"],
            "join_key": "document_id",
            "write_scope": "RESULT",
            "verify_scope": "RESULT",
        },
    }


def _bindings(source_ref="U1"):
    return [
        {"set_id": "A", "read_api_reason": "返回文件夹内文档，产出document且没有增加标星条件。",
         "read_api": "drive.list_folder_documents", "source_ref": source_ref,
         "requested_scope": "指定文件夹内的全部文档",
         "coverage_reason": "分页读完才能覆盖全部文档。", "completion_condition": "读到没有下一页"},
        {"set_id": "B", "read_api_reason": "返回已标星文档，产出document且没有增加文件夹条件。",
         "read_api": "drive.list_starred_documents", "source_ref": source_ref,
         "requested_scope": "全部已标星文档",
         "coverage_reason": "分页读完才能覆盖全部标星文档。", "completion_condition": "读到没有下一页"},
    ]


def _binding_checks(assessment="COMPLETE_MATCH"):
    return [
        {"reason": "A的描述与集合定义一致且已经读完。", "assessment": assessment, "set_id": "A"},
        {"reason": "B的描述与集合定义一致且已经读完。", "assessment": "COMPLETE_MATCH", "set_id": "B"},
    ]


def test_target_read_requires_explicit_phase_complete_bindings_and_real_calls():
    execute = make_execute_tool(World())
    state = _target_selection_state()
    good = {
        "source_refs": ["U1"],
        "reason": "接口与范围来自当前Step；参数来自用户请求。",
        "action_phase": "TARGET_READ",
        "set_bindings": _bindings(),
        "code": (
            "a = apis.drive.list_folder_documents(folder_id='known')\n"
            "b = apis.drive.list_starred_documents()\nprint(a, b)"
        ),
    }
    grounded = validate_and_strip_tool_call(
        Request({"name": "appworld_execute", "id": "x", "args": good}, execute, state),
        {},
    )
    assert grounded.tool_call["args"]["action_phase"] == "TARGET_READ"

    for change, message in [
        ({"action_phase": None}, "explicit valid action_phase"),
        ({"set_bindings": _bindings()[:1]}, "cover every target_selection operand"),
        ({"code": "print(apis.drive.list_starred_documents())"},
         "current code does not call"),
    ]:
        bad = {**good, **change}
        with pytest.raises(ValueError, match=message):
            validate_and_strip_tool_call(
                Request({"name": "appworld_execute", "id": "x", "args": bad},
                        execute, state),
                {},
            )


def test_prerequisite_does_not_claim_target_sets_or_complete_task():
    execute = make_execute_tool(World())
    state = _target_selection_state()
    valid = {
        "source_refs": ["U1"], "reason": "获取任务前置条件",
        "action_phase": "PREREQUISITE", "set_bindings": [], "code": "print(1)",
    }
    validate_and_strip_tool_call(
        Request({"name": "appworld_execute", "id": "x", "args": valid},
                execute, state),
        {},
    )
    with pytest.raises(ValueError, match="must not claim target-set bindings"):
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "x",
                     "args": {**valid, "set_bindings": _bindings()}},
                    execute, state),
            {},
        )
    with pytest.raises(ValueError, match="complete_task is allowed only"):
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "x",
                     "args": {**valid, "code":
                         "print(apis.supervisor.complete_task(answer='done'))"}},
                    execute, state),
            {},
        )


def test_harness_normalizes_nullish_action_card_values_before_validation():
    execute = make_execute_tool(World())
    state = _target_selection_state()
    for sentinel in (None, False, "", "null", "NULL", "none", "false"):
        args = {
            "source_refs": ["U1"],
            "reason": "读取登录前置信息",
            "action_phase": "PREREQUISITE",
            "binding_ref": sentinel,
            "binding_checks": sentinel,
            "set_bindings": sentinel,
            "code": "print(1)",
        }
        grounded = validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "x", "args": args},
                    execute, state),
            {},
        )
        assert grounded.tool_call["args"]["binding_ref"] is None
        assert grounded.tool_call["args"]["binding_checks"] == []
        assert grounded.tool_call["args"]["set_bindings"] == []


def test_target_write_reuses_prior_successful_set_read_result():
    execute = make_execute_tool(World())
    read_args = {
        "source_refs": ["U1"], "reason": "读取两个集合",
        "action_phase": "TARGET_READ", "set_bindings": _bindings(),
        "code": (
            "a = apis.drive.list_folder_documents(folder_id='known')\n"
            "b = apis.drive.list_starred_documents()\nprint(a, b)"
        ),
    }
    messages = [
        HumanMessage(content="处理文件夹内且已标星的文档"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "raw-read", "args": read_args}
        ]),
        ToolMessage(content="real rows", tool_call_id="raw-read",
                    name="appworld_execute"),
    ]
    state = _target_selection_state(messages)
    write_args = {
        "source_refs": ["E1"], "reason": "目标ID来自已成功读取的两个集合。",
        "action_phase": "TARGET_WRITE", "binding_ref": "E1",
        "binding_checks": _binding_checks(), "set_bindings": [],
        "code": "print(apis.drive.archive_documents(document_ids=result_ids))",
    }
    grounded = validate_and_strip_tool_call(
        Request({"name": "appworld_execute", "id": "write", "args": write_args},
                execute, state),
        {"raw-read": "E1"},
    )
    assert grounded.tool_call["args"]["set_bindings"] == _bindings("E1")

    wrong = {**write_args, "set_bindings": _bindings("E1")}
    with pytest.raises(ValueError, match="must not recopy set_bindings"):
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "write", "args": wrong},
                    execute, state),
            {"raw-read": "E1"},
        )


def test_finalize_requires_matching_read_write_verify_chain_and_verify_reference():
    execute = make_execute_tool(World())
    read_args = {
        "source_refs": ["U1"], "reason": "读取完整目标集合",
        "action_phase": "TARGET_READ", "set_bindings": _bindings(),
        "code": (
            "a = apis.drive.list_folder_documents(folder_id='known')\n"
            "b = apis.drive.list_starred_documents()\nprint(a, b)"
        ),
    }
    write_args = {
        "source_refs": ["E1"], "reason": "只处理已读取集合的交集",
        "action_phase": "TARGET_WRITE", "binding_ref": "E1",
        "binding_checks": _binding_checks(), "set_bindings": [],
        "code": "print(apis.drive.archive_documents(document_ids=result_ids))",
    }
    verify_args = {
        "source_refs": ["E1"], "reason": "回读同一目标集合",
        "action_phase": "TARGET_VERIFY", "binding_ref": "E1", "set_bindings": [],
        "code": "print(apis.drive.show_documents(document_ids=result_ids))",
    }
    messages = [
        HumanMessage(content="处理文件夹内且已标星的文档"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "raw-read", "args": read_args}]),
        ToolMessage(content="read rows", tool_call_id="raw-read",
                    name="appworld_execute"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "raw-write", "args": write_args}]),
        ToolMessage(content="write ok", tool_call_id="raw-write",
                    name="appworld_execute"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "raw-verify", "args": verify_args}]),
        ToolMessage(content="verified rows", tool_call_id="raw-verify",
                    name="appworld_execute"),
    ]
    state = _target_selection_state(messages)
    evidence = {"raw-read": "E1", "raw-write": "E2", "raw-verify": "E3"}
    final_args = {
        "source_refs": ["E3"], "reason": "引用成功回读结果后结案",
        "action_phase": "FINALIZE", "set_bindings": [],
        "code": "print(apis.supervisor.complete_task(answer='done'))",
    }
    validate_and_strip_tool_call(
        Request({"name": "appworld_execute", "id": "final", "args": final_args},
                execute, state),
        evidence,
    )

    with pytest.raises(ValueError, match="must cite the E reference"):
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "final",
                     "args": {**final_args, "source_refs": ["E2"]}},
                    execute, state),
            evidence,
        )


def test_finalize_rejects_missing_or_mismatched_phase_chain():
    execute = make_execute_tool(World())
    state = _target_selection_state()
    final_args = {
        "source_refs": ["U1"], "reason": "错误地提前结案",
        "action_phase": "FINALIZE", "set_bindings": [],
        "code": "print(apis.supervisor.complete_task(answer='done'))",
    }
    with pytest.raises(ValueError, match="prior successful target phases"):
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "final",
                     "args": final_args}, execute, state),
            {},
        )


def test_finalize_rejects_verify_against_a_different_dynamic_read_receipt():
    execute = make_execute_tool(World())
    read_args = {
        "source_refs": ["U1"], "reason": "读取当时的完整目标集合",
        "action_phase": "TARGET_READ", "set_bindings": _bindings(),
        "code": (
            "a = apis.drive.list_folder_documents(folder_id='known')\n"
            "b = apis.drive.list_starred_documents()\nprint(a, b)"
        ),
    }
    write_args = {
        "source_refs": ["E1"], "reason": "按第一次冻结名单写入",
        "action_phase": "TARGET_WRITE", "binding_ref": "E1",
        "binding_checks": _binding_checks(), "set_bindings": [],
        "code": "print(apis.drive.archive_documents(document_ids=result_ids))",
    }
    verify_args = {
        "source_refs": ["E3"], "reason": "错误地按第二次查询名单回读",
        "action_phase": "TARGET_VERIFY", "binding_ref": "E3", "set_bindings": [],
        "code": "print(apis.drive.show_documents(document_ids=result_ids))",
    }
    messages = [
        HumanMessage(content="处理文件夹内且已标星的文档"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "read-one", "args": read_args}]),
        ToolMessage(content="first snapshot", tool_call_id="read-one",
                    name="appworld_execute"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "write", "args": write_args}]),
        ToolMessage(content="write ok", tool_call_id="write",
                    name="appworld_execute"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "read-two", "args": read_args}]),
        ToolMessage(content="second snapshot", tool_call_id="read-two",
                    name="appworld_execute"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "verify", "args": verify_args}]),
        ToolMessage(content="verify result", tool_call_id="verify",
                    name="appworld_execute"),
    ]
    state = _target_selection_state(messages)
    evidence = {
        "read-one": "E1", "write": "E2", "read-two": "E3", "verify": "E4",
    }
    final_args = {
        "source_refs": ["E4"], "reason": "尝试按不同名单结案",
        "action_phase": "FINALIZE", "set_bindings": [],
        "code": "print(apis.supervisor.complete_task(answer='done'))",
    }
    with pytest.raises(ValueError, match="same dynamic TARGET_READ receipt"):
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "final",
                     "args": final_args}, execute, state),
            evidence,
        )


def test_action_card_rejection_gives_specific_correction_and_expected_lineage():
    execute = make_execute_tool(World())
    state = _target_selection_state()
    bad_read = {
        "source_refs": ["U1"], "reason": "读取范围",
        "action_phase": "TARGET_READ", "set_bindings": _bindings()[:1],
        "code": "print(apis.drive.list_folder_documents(folder_id='known'))",
    }
    with pytest.raises(ValueError) as caught:
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "bad-read", "args": bad_read},
                    execute, state),
            {},
        )
    message = str(caught.value)
    assert "ACTION_CARD_REJECTED[BINDINGS_COVERAGE]" in message
    assert "missing=['B']" in message
    assert "set_id/read_api_reason/read_api/source_ref/requested_scope/coverage_reason/completion_condition" in message
    assert "不要原样重复" in message

    read_args = {
        "source_refs": ["U1"], "reason": "读取两个集合",
        "action_phase": "TARGET_READ", "set_bindings": _bindings(),
        "code": (
            "a = apis.drive.list_folder_documents(folder_id='known')\n"
            "b = apis.drive.list_starred_documents()\nprint(a, b)"
        ),
    }
    messages = [
        HumanMessage(content="处理文件夹内且已标星的文档"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "raw-read", "args": read_args}
        ]),
        ToolMessage(content="real rows", tool_call_id="raw-read",
                    name="appworld_execute"),
    ]
    state = _target_selection_state(messages)
    wrong_verify = {
        "source_refs": ["E1"], "reason": "错误地重复抄写绑定",
        "action_phase": "TARGET_VERIFY", "binding_ref": "E1",
        "set_bindings": _bindings("E1"),
        "code": "print(apis.drive.show_documents(document_ids=result_ids))",
    }
    with pytest.raises(ValueError) as caught:
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "bad-verify",
                     "args": wrong_verify}, execute, state),
            {"raw-read": "E1"},
        )
    message = str(caught.value)
    assert "ACTION_CARD_REJECTED[TARGET_RECEIPT_DUPLICATED]" in message
    assert "set_bindings: []" in message


def test_read_only_target_can_finalize_after_read_and_cannot_write():
    execute = make_execute_tool(World())
    read_args = {
        "source_refs": ["U1"], "reason": "读取两个只读集合",
        "action_phase": "TARGET_READ", "set_bindings": _bindings(),
        "code": (
            "a = apis.drive.list_folder_documents(folder_id='known')\n"
            "b = apis.drive.list_starred_documents()\nprint(a, b)"
        ),
    }
    messages = [
        HumanMessage(content="告诉我文件夹内且已标星的文档"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "raw-read", "args": read_args}]),
        ToolMessage(content="read rows", tool_call_id="raw-read",
                    name="appworld_execute"),
    ]
    state = _target_selection_state(messages, effect_mode="READ_ONLY")
    final_args = {
        "source_refs": ["E1"], "reason": "读取结果足以回答只读任务",
        "action_phase": "FINALIZE", "binding_checks": [], "set_bindings": [],
        "code": "print(apis.supervisor.complete_task(answer='done'))",
    }
    validate_and_strip_tool_call(
        Request({"name": "appworld_execute", "id": "final", "args": final_args},
                execute, state),
        {"raw-read": "E1"},
    )
    write_args = {
        "source_refs": ["E1"], "reason": "不应修改只读任务",
        "action_phase": "TARGET_WRITE", "binding_ref": "E1",
        "binding_checks": _binding_checks(), "set_bindings": [],
        "code": "print(apis.drive.archive_documents(document_ids=result_ids))",
    }
    with pytest.raises(ValueError, match="READ_ONLY"):
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "write", "args": write_args},
                    execute, state),
            {"raw-read": "E1"},
        )


def test_different_binding_cancels_write_without_running_world_and_supersedes_receipt():
    from workers.evidence_refs import EvidenceReferenceMiddleware

    world = World()
    execute = make_execute_tool(world)
    read_args = {
        "source_refs": ["U1"], "reason": "读取两个集合",
        "action_phase": "TARGET_READ", "set_bindings": _bindings(),
        "code": (
            "a = apis.drive.list_folder_documents(folder_id='known')\n"
            "b = apis.drive.list_starred_documents()\nprint(a, b)"
        ),
    }
    messages = [
        HumanMessage(content="处理文件夹内且已标星的文档"),
        AIMessage(content="", tool_calls=[
            {"name": "appworld_execute", "id": "raw-read", "args": read_args}]),
        ToolMessage(content="read rows", tool_call_id="raw-read",
                    name="appworld_execute"),
    ]
    state = _target_selection_state(messages)
    write_args = {
        "source_refs": ["E1"], "reason": "复核后发现A不一致，所以不应写入",
        "action_phase": "TARGET_WRITE", "binding_ref": "E1",
        "binding_checks": _binding_checks("INCOMPLETE_RESULT"), "set_bindings": [],
        "code": "print(apis.drive.archive_documents(document_ids=result_ids))",
    }
    request = Request(
        {"name": "appworld_execute", "id": "write", "args": write_args},
        execute,
        state,
    )
    converted = validate_and_strip_tool_call(request, {"raw-read": "E1"})
    assert ACTION_CARD_REREAD_MARKER in converted.tool_call["args"]

    called = []
    result = EvidenceReferenceMiddleware().wrap_tool_call(
        request, lambda req: called.append(req) or ToolMessage(
            content="should not run", tool_call_id="write", name="appworld_execute"))
    assert called == []
    assert "代码未执行" in str(result.update["messages"][0].content)
    assert result.update["worker_invalid_target_receipts"] == ["E1"]

    invalid_state = {**state, "worker_invalid_target_receipts": ["E1"]}
    same_args = {**write_args, "binding_checks": _binding_checks("COMPLETE_MATCH")}
    with pytest.raises(ValueError, match="SUPERSEDED"):
        validate_and_strip_tool_call(
            Request({"name": "appworld_execute", "id": "again", "args": same_args},
                    execute, invalid_state),
            {"raw-read": "E1"},
        )


def test_target_read_review_shows_exact_source_description_for_mutation_only():
    rag = HumanMessage(
        content=(
            "drive.list_folder_documents: Return documents contained in the selected folder. "
            "drive.list_starred_documents: Return documents starred by the current user."
        ),
        additional_kwargs={"knowledge_source": True},
    )
    state = _target_selection_state([HumanMessage(content="task"), rag])
    args = {
        "source_refs": ["R1"], "reason": "接口来自目录",
        "action_phase": "TARGET_READ",
        "set_bindings": _bindings("R1"),
        "code": (
            "a = apis.drive.list_folder_documents(folder_id='known')\n"
            "b = apis.drive.list_starred_documents()\nprint(a, b)"
        ),
    }
    prompt = target_read_binding_review(
        Request({"name": "appworld_execute", "id": "read", "args": args},
                make_execute_tool(World()), state),
        {},
    )
    assert "TARGET_BINDING_REVIEW_REQUIRED" in prompt
    assert "Return documents contained in the selected folder" in prompt
    assert prompt.index("reason") < prompt.index("assessment")

    read_only_state = _target_selection_state(
        [HumanMessage(content="task"), rag], effect_mode="READ_ONLY")
    assert target_read_binding_review(
        Request({"name": "appworld_execute", "id": "read", "args": args},
                make_execute_tool(World()), read_only_state),
        {},
    ) == ""
