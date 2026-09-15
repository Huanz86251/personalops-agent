"""Provider-free contracts for structured target selection in Scheduler plans."""

import pytest
from pydantic import ValidationError

from planning_models import PlanStep


def intersection_step():
    return PlanStep.model_validate({
        "step_id": 1,
        "objective": "归档我创建的文件夹中、我标星的文档",
        "success_criteria": ["交集中的文档均已归档"],
        "target_selection": {
            "target_entity": "document",
            "sets": [
                {
                    "set_id": "A",
                    "definition": "我创建的文件夹中的文档",
                    "condition_owner": "folder",
                    "result_entity": "document",
                },
                {
                    "set_id": "B",
                    "definition": "我标星的文档",
                    "condition_owner": "document",
                    "result_entity": "document",
                },
            ],
            "operation": "INTERSECTION",
            "operands": ["A", "B"],
            "join_key": "document_id",
        },
    })


def test_intersection_contract_is_machine_readable_and_write_bounded():
    selection = intersection_step().target_selection
    assert selection is not None
    assert selection.operation == "INTERSECTION"
    assert selection.operands == ["A", "B"]
    assert selection.join_key == "document_id"
    assert selection.write_scope == "RESULT"
    assert selection.verify_scope == "RESULT"


@pytest.mark.parametrize(
    "change",
    [
        {"join_key": None},
        {"operands": ["A"]},
        {"operands": ["A", "A"]},
    ],
)
def test_multiset_expression_rejects_missing_or_invalid_operands(change):
    payload = intersection_step().model_dump()
    payload["target_selection"].update(change)
    with pytest.raises(ValidationError):
        PlanStep.model_validate(payload)


def test_expression_rejects_moving_one_set_to_a_different_result_entity():
    payload = intersection_step().model_dump()
    payload["target_selection"]["sets"][0]["result_entity"] = "folder"
    with pytest.raises(ValidationError, match="target_entity"):
        PlanStep.model_validate(payload)


def test_direct_selection_stays_optional_for_simple_steps():
    simple = PlanStep(
        step_id=1,
        objective="读取当前时间",
        success_criteria=["返回真实时间"],
    )
    assert simple.target_selection is None


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (True, "ENABLED"),
        ("true", "ENABLED"),
        (False, "DISABLED"),
        ("false", "DISABLED"),
        ("unexpected", "ENABLED"),
    ],
)
def test_plan_step_tool_access_accepts_boolean_strings(raw_value, expected):
    step = PlanStep(
        step_id=1,
        objective="完成当前任务",
        success_criteria=["已完成"],
        tool_access=raw_value,
    )
    assert step.tool_access == expected


def test_plan_step_tool_access_defaults_open():
    step = PlanStep(
        step_id=1,
        objective="完成当前任务",
        success_criteria=["已完成"],
    )
    assert step.tool_access == "ENABLED"
