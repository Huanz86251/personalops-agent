from __future__ import annotations

import json
import logging
import copy
from collections.abc import (
    Mapping,
)

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
)

from observability import (
    set_span_attributes,
    set_span_output,
    trace_span,
)

from planning_models import (
    PlanStep,
    StepCriterionResult,
    StepReport,
)
from reporting.models import (
    StepReviewPacket,
)
from reporting.criteria import ReferencedStepReport, normalize_report
from schema_utils import is_schema_repairable_error, schema_repair_feedback

from prompt_loader import (
    render_prompt,
)


logger = logging.getLogger(
    "agent"
)


# StepReport格式错误时，最多额外修复三次。
MAX_STEP_REPORT_RETRIES = 3
# Reporter内部使用的简单上下文预算启发式。
#
# 当CLOUD_LLM_MAX_TOKENS=5000时：
#
# total_budget = 5000 * 3 = 15000
# output_reserve = 15000 / 3 = 5000
#
# 剩余部分用于固定Prompt、JSON Schema和执行轨迹。
STEP_REPORT_TOTAL_BUDGET_MULTIPLIER = 3
STEP_REPORT_OUTPUT_RESERVE_DIVISOR = 3

# 防止估算误差刚好把上下文塞满。
STEP_REPORT_FIXED_SAFETY_TOKENS = 256

# 即使固定Prompt较长，
# 也至少给执行轨迹保留少量空间。
STEP_REPORT_MIN_TRACE_TOKENS = 512

# Reporter最终fallback保留多少原始轨迹。
STEP_REPORT_FALLBACK_TRACE_TOKENS = 800

@dataclass(
    frozen=True,
)
class StepReportRunResult:
    """一次StepReport生成结果。"""

    report: StepReport

    model_rounds_used: int

    used_fallback: bool
    skill_snapshot: dict | None = None


def _make_json_safe(
    value: Any,
) -> Any:
    """把常见对象转换成可以稳定写入Prompt的结构。"""

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(
        model_dump
    ):
        try:
            return _make_json_safe(
                model_dump(
                    mode="json",
                )
            )

        except Exception:
            pass

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): (
                _make_json_safe(
                    item
                )
            )

            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            _make_json_safe(
                item
            )

            for item in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    return str(
        value
    )


def _to_prompt_text(
    value: Any,
    empty_text: str,
) -> str:
    """把Python对象转换成Prompt文字。"""

    if value is None:
        return empty_text

    if isinstance(
        value,
        str,
    ):
        return (
            value.strip()
            or empty_text
        )

    return json.dumps(
        _make_json_safe(
            value
        ),

        ensure_ascii=False,

        indent=2,

        default=str,
    )


def _estimate_tokens(
    text: str,
) -> int:
    """保守估算中英文混合文本Token。

    这里只用于Step Reporter输入裁剪，
    不用于供应商计费或精确上下文判断。

    非ASCII字符按照约1字符1Token估算；
    ASCII、英文和代码按照约4字符1Token估算。
    """

    ascii_chars = sum(
        ord(
            character
        ) < 128

        for character
        in text
    )

    non_ascii_chars = (
        len(
            text
        )
        - ascii_chars
    )

    return (
        non_ascii_chars
        + (
            ascii_chars
            + 3
        )
        // 4
    )


def _clip_text_to_token_budget(
    text: str,
    *,
    max_tokens: int,
    tail_only: bool,
) -> str:
    """把一段文字裁剪到指定Token预算。

    tail_only=True：
        用于ToolMessage和工具参数，
        主要保留后半部分。

    tail_only=False：
        用于Assistant文字，
        同时保留头部和尾部。
    """

    normalized_text = str(
        text
    )

    if max_tokens < 1:
        return ""

    if (
        _estimate_tokens(
            normalized_text
        )
        <= max_tokens
    ):
        return normalized_text

    marker = (
        "……前部因Reporter上下文限制已省略……\n"

        if tail_only

        else (
            "\n……中部因Reporter上下文限制已省略……\n"
        )
    )

    if (
        _estimate_tokens(
            marker
        )
        >= max_tokens
    ):
        marker = "……\n"

    low = 0
    high = len(
        normalized_text
    )

    while low < high:
        kept_chars = (
            low
            + high
            + 1
        ) // 2

        if tail_only:
            candidate = (
                marker
                + normalized_text[
                    -kept_chars:
                ]
            )

        else:
            head_chars = (
                kept_chars
                // 2
            )

            tail_chars = (
                kept_chars
                - head_chars
            )

            candidate = (
                normalized_text[
                    :head_chars
                ]
                + marker
                + normalized_text[
                    -tail_chars:
                ]
            )

        if (
            _estimate_tokens(
                candidate
            )
            <= max_tokens
        ):
            low = kept_chars

        else:
            high = (
                kept_chars
                - 1
            )

    if tail_only:
        return (
            marker
            + normalized_text[
                -low:
            ]
        )

    head_chars = (
        low
        // 2
    )

    tail_chars = (
        low
        - head_chars
    )

    return (
        normalized_text[
            :head_chars
        ]
        + marker
        + normalized_text[
            -tail_chars:
        ]
    )


def _message_to_report_item(
    message: Any,
) -> dict[
    str,
    Any,
]:
    """把LangChain消息转换成Reporter使用的短结构。"""

    if isinstance(
        message,
        Mapping,
    ):
        role = (
            message.get(
                "role"
            )
            or message.get(
                "type"
            )
            or ""
        )

        content = message.get(
            "content",
            "",
        )

        name = (
            message.get(
                "name"
            )
            or ""
        )

        tool_call_id = (
            message.get(
                "tool_call_id"
            )
            or ""
        )

        tool_calls = (
            message.get(
                "tool_calls"
            )
            or []
        )

    else:
        role = (
            getattr(
                message,
                "type",
                "",
            )
            or getattr(
                message,
                "role",
                "",
            )
            or ""
        )

        content = getattr(
            message,
            "content",
            "",
        )

        name = (
            getattr(
                message,
                "name",
                "",
            )
            or ""
        )

        tool_call_id = (
            getattr(
                message,
                "tool_call_id",
                "",
            )
            or ""
        )

        tool_calls = (
            getattr(
                message,
                "tool_calls",
                None,
            )
            or []
        )

    normalized_role = str(
        role
    ).strip().lower()

    if normalized_role == "ai":
        normalized_role = "assistant"

    elif normalized_role == "human":
        normalized_role = "user"

    if isinstance(
        content,
        list,
    ):
        content_parts: list[
            str
        ] = []

        for item in content:
            if isinstance(
                item,
                str,
            ):
                content_parts.append(
                    item
                )

            elif (
                isinstance(
                    item,
                    Mapping,
                )
                and isinstance(
                    item.get(
                        "text"
                    ),
                    str,
                )
            ):
                content_parts.append(
                    item[
                        "text"
                    ]
                )

        content_text = "".join(
            content_parts
        )

    elif content is None:
        content_text = ""

    else:
        content_text = str(
            content
        )

    if normalized_role == "user":
        # 当前用户请求、整体目标、当前Step和成功标准
        # 已经在Reporter Prompt中分别提供。
        #
        # 这里的UserMessage通常只是重复的巨大Step指令。
        return {
            "role": "user",

            "content": (
                "[当前Step指令已在Prompt其他区块中提供，"
                "此处省略重复内容。]"
            ),
        }

    item: dict[
        str,
        Any,
    ] = {
        "role": (
            normalized_role
            or "unknown"
        ),

        "content": (
            content_text
        ),
    }

    if normalized_role == "tool":
        item[
            "tool_name"
        ] = str(
            name
        ).strip()

        item[
            "tool_call_id"
        ] = str(
            tool_call_id
        ).strip()

        return item

    if (
        normalized_role
        != "assistant"

        or not isinstance(
            tool_calls,
            list,
        )
    ):
        return item

    normalized_calls: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for tool_call in tool_calls:
        if isinstance(
            tool_call,
            Mapping,
        ):
            function_data = tool_call.get(
                "function"
            )

            if not isinstance(
                function_data,
                Mapping,
            ):
                function_data = {}

            tool_name = (
                tool_call.get(
                    "name"
                )
                or function_data.get(
                    "name"
                )
                or ""
            )

            arguments = tool_call.get(
                "args"
            )

            if arguments is None:
                arguments = tool_call.get(
                    "arguments"
                )

            if arguments is None:
                arguments = function_data.get(
                    "arguments"
                )

            call_id = (
                tool_call.get(
                    "id"
                )
                or tool_call.get(
                    "tool_call_id"
                )
                or ""
            )

            call_type = (
                tool_call.get(
                    "type"
                )
                or ""
            )

        else:
            tool_name = (
                getattr(
                    tool_call,
                    "name",
                    "",
                )
                or ""
            )

            arguments = getattr(
                tool_call,
                "args",
                None,
            )

            if arguments is None:
                arguments = getattr(
                    tool_call,
                    "arguments",
                    None,
                )

            call_id = (
                getattr(
                    tool_call,
                    "id",
                    "",
                )
                or getattr(
                    tool_call,
                    "tool_call_id",
                    "",
                )
                or ""
            )

            call_type = (
                getattr(
                    tool_call,
                    "type",
                    "",
                )
                or ""
            )

        if arguments is None:
            arguments_text = ""

        elif isinstance(
            arguments,
            str,
        ):
            arguments_text = arguments

        else:
            arguments_text = json.dumps(
                _make_json_safe(
                    arguments
                ),

                ensure_ascii=False,

                separators=(
                    ",",
                    ":",
                ),

                default=str,
            )

        normalized_calls.append(
            {
                "tool_call_id": str(
                    call_id
                ).strip(),

                "tool_name": str(
                    tool_name
                ).strip(),

                "arguments": (
                    arguments_text
                ),

                "type": str(
                    call_type
                ).strip(),
            }
        )

    if normalized_calls:
        item[
            "tool_calls"
        ] = normalized_calls

    return item


def _compact_step_execution_trace(
    step_execution_trace: Any,
    *,
    max_tokens: int,
) -> tuple[
    Any,
    bool,
    int,
    int,
]:
    """仅在超限时压缩Reporter看到的执行轨迹。

    Returns:
        compacted_trace：
            最终提供给Reporter的轨迹。

        trace_was_compacted：
            是否发生了压缩。

        original_tokens：
            原始轨迹估算Token。

        compacted_tokens：
            压缩后轨迹估算Token。
    """

    if max_tokens < 1:
        raise ValueError(
            "max_tokens不能小于1。"
        )

    original_text = _to_prompt_text(
        step_execution_trace,
        "没有可用执行轨迹。",
    )

    original_tokens = _estimate_tokens(
        original_text
    )

    # 普通小任务不做任何损失性处理。
    if original_tokens <= max_tokens:
        return (
            step_execution_trace,
            False,
            original_tokens,
            original_tokens,
        )

    if not isinstance(
        step_execution_trace,
        Mapping,
    ):
        normalized_trace: dict[
            str,
            Any,
        ] = {
            "trace": (
                _make_json_safe(
                    step_execution_trace
                )
            )
        }

    else:
        normalized_trace = {}

        for key, value in (
            step_execution_trace.items()
        ):
            normalized_key = str(
                key
            )

            if (
                normalized_key
                != "current_attempt"

                or not isinstance(
                    value,
                    Mapping,
                )
            ):
                normalized_trace[
                    normalized_key
                ] = _make_json_safe(
                    value
                )

                continue

            normalized_attempt: dict[
                str,
                Any,
            ] = {}

            for attempt_key, attempt_value in (
                value.items()
            ):
                normalized_attempt_key = str(
                    attempt_key
                )

                if normalized_attempt_key == "messages":
                    normalized_attempt[
                        "messages"
                    ] = (
                        [
                            _message_to_report_item(
                                message
                            )

                            for message
                            in attempt_value
                        ]

                        if isinstance(
                            attempt_value,
                            list,
                        )

                        else []
                    )

                    continue

                if (
                    normalized_attempt_key
                    == "execution_summary"

                    and isinstance(
                        attempt_value,
                        Mapping,
                    )
                ):
                    summary = {
                        str(summary_key): (
                            _make_json_safe(
                                summary_value
                            )
                        )

                        for summary_key, summary_value
                        in attempt_value.items()

                        if str(
                            summary_key
                        ) != "timeline"
                    }

                    summary[
                        "timeline_omitted"
                    ] = (
                        "消息已单独保留，"
                        "因此删除重复timeline。"
                    )

                    normalized_attempt[
                        normalized_attempt_key
                    ] = summary

                    continue

                normalized_attempt[
                    normalized_attempt_key
                ] = _make_json_safe(
                    attempt_value
                )

            normalized_trace[
                "current_attempt"
            ] = normalized_attempt

    compacted_trace = copy.deepcopy(
        normalized_trace
    )

    compacted_trace[
        "trace_compression"
    ] = {
        "applied": True,

        "original_estimated_tokens": (
            original_tokens
        ),

        "max_trace_tokens": (
            max_tokens
        ),

        "strategy": (
            "保留消息顺序和Assistant消息条目；"
            "工具结果与工具参数优先保留尾部；"
            "删除重复用户Step指令和重复timeline。"
        ),
    }

    current_attempt = compacted_trace.get(
        "current_attempt"
    )

    messages = (
        current_attempt.get(
            "messages"
        )

        if isinstance(
            current_attempt,
            dict,
        )

        else None
    )

    if not isinstance(
        messages,
        list,
    ):
        compacted_text = _to_prompt_text(
            compacted_trace,
            "没有可用执行轨迹。",
        )

        compacted_tokens = _estimate_tokens(
            compacted_text
        )

        if compacted_tokens <= max_tokens:
            return (
                compacted_trace,
                True,
                original_tokens,
                compacted_tokens,
            )

        fallback_trace = {
            "trace_compression": {
                "applied": True,
                "original_estimated_tokens": (
                    original_tokens
                ),
                "max_trace_tokens": (
                    max_tokens
                ),
                "strategy": (
                    "非消息轨迹仍然超限，"
                    "最终只保留轨迹尾部。"
                ),
            },

            "trace_tail": (
                _clip_text_to_token_budget(
                    compacted_text,
                    max_tokens=max_tokens,
                    tail_only=True,
                )
            ),
        }

        return (
            fallback_trace,
            True,
            original_tokens,
            _estimate_tokens(
                _to_prompt_text(
                    fallback_trace,
                    "",
                )
            ),
        )

    assistant_indices = [
        index

        for index, item
        in enumerate(
            messages
        )

        if (
            isinstance(
                item,
                dict,
            )
            and item.get(
                "role"
            ) == "assistant"
        )
    ]

    final_assistant_index = (
        assistant_indices[-1]

        if assistant_indices

        else None
    )

    # 每个slot包含：
    #
    # container：
    #     需要修改的字典。
    #
    # key：
    #     content或arguments。
    #
    # text：
    #     原始文字。
    #
    # tail_only：
    #     是否只保留尾部。
    #
    # weight：
    #     在总剩余预算中的相对权重。
    slots: list[
        tuple[
            dict[
                str,
                Any,
            ],
            str,
            str,
            bool,
            int,
        ]
    ] = []

    for message_index, item in enumerate(
        messages
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        role = str(
            item.get(
                "role",
                "",
            )
        )

        content = item.get(
            "content"
        )

        if (
            isinstance(
                content,
                str,
            )
            and role != "user"
        ):
            if role == "assistant":
                weight = (
                    6

                    if (
                        message_index
                        == final_assistant_index
                    )

                    else 4
                )

                tail_only = False

            elif role == "tool":
                weight = 3
                tail_only = True

            else:
                weight = 1
                tail_only = False

            slots.append(
                (
                    item,
                    "content",
                    content,
                    tail_only,
                    weight,
                )
            )

        tool_calls = item.get(
            "tool_calls"
        )

        if isinstance(
            tool_calls,
            list,
        ):
            for tool_call in tool_calls:
                if (
                    isinstance(
                        tool_call,
                        dict,
                    )
                    and isinstance(
                        tool_call.get(
                            "arguments"
                        ),
                        str,
                    )
                ):
                    slots.append(
                        (
                            tool_call,
                            "arguments",
                            tool_call[
                                "arguments"
                            ],
                            True,
                            2,
                        )
                    )

    original_slot_values = [
        slot[2]
        for slot in slots
    ]

    # 先把可裁剪内容置空，
    # 算出JSON结构和元数据本身占多少Token。
    for (
        container,
        key,
        _text,
        _tail_only,
        _weight,
    ) in slots:
        container[
            key
        ] = ""

    base_tokens = _estimate_tokens(
        _to_prompt_text(
            compacted_trace,
            "",
        )
    )

    payload_tokens = max(
        0,
        max_tokens
        - base_tokens,
    )

    total_weight = max(
        1,
        sum(
            slot[4]
            for slot in slots
        ),
    )

    # 最多做四轮确定性收紧。
    # 正常情况下第一轮即可进入预算。
    for shrink_round in range(
        4
    ):
        round_payload = int(
            payload_tokens
            * (
                0.78
                ** shrink_round
            )
        )

        for slot_index, (
            container,
            key,
            _text,
            tail_only,
            weight,
        ) in enumerate(
            slots
        ):
            slot_budget = (
                round_payload
                * weight
                // total_weight
            )

            container[
                key
            ] = _clip_text_to_token_budget(
                original_slot_values[
                    slot_index
                ],

                max_tokens=(
                    slot_budget
                ),

                tail_only=(
                    tail_only
                ),
            )

        compacted_tokens = _estimate_tokens(
            _to_prompt_text(
                compacted_trace,
                "",
            )
        )

        if compacted_tokens <= max_tokens:
            compacted_trace[
                "trace_compression"
            ][
                "compacted_estimated_tokens"
            ] = compacted_tokens

            return (
                compacted_trace,
                True,
                original_tokens,
                compacted_tokens,
            )

    # 极端情况下仍超限，
    # 最终保留规范化轨迹尾部。
    fallback_text = _to_prompt_text(
        compacted_trace,
        "没有可用执行轨迹。",
    )

    fallback_trace = {
        "trace_compression": {
            "applied": True,

            "original_estimated_tokens": (
                original_tokens
            ),

            "max_trace_tokens": (
                max_tokens
            ),

            "strategy": (
                "规范化后仍超限，"
                "最终只保留压缩轨迹尾部。"
            ),
        },

        "trace_tail": (
            _clip_text_to_token_budget(
                fallback_text,

                max_tokens=(
                    max_tokens
                ),

                tail_only=True,
            )
        ),
    }

    fallback_tokens = _estimate_tokens(
        _to_prompt_text(
            fallback_trace,
            "",
        )
    )

    return (
        fallback_trace,
        True,
        original_tokens,
        fallback_tokens,
    )

def _build_report_fallback(
    *,
    current_step: PlanStep,
    stop_reason: str,
    error: str,
    step_execution_trace: Any,
) -> StepReport:
    """Reporter失败时保留轨迹尾部，但不伪造成功结论。"""

    trace_tail = (
        _clip_text_to_token_budget(
            _to_prompt_text(
                step_execution_trace,
                "没有可用执行轨迹。",
            ),

            max_tokens=(
                STEP_REPORT_FALLBACK_TRACE_TOKENS
            ),

            tail_only=True,
        )
    )

    return StepReport(
        step_id=(
            current_step.step_id
        ),

        status="PARTIAL",

        summary=(
            "当前Step已经停止执行，"
            "但Reporter没有生成合法的"
            "结构化StepReport。"
        ),

        stop_reason=(
            stop_reason
            or "StepReport生成失败。"
        ),

        criterion_results=[
            StepCriterionResult(
                criterion=(
                    criterion
                ),

                status="UNKNOWN",

                evidence=[],
            )

            for criterion
            in current_step.success_criteria
        ],

        confirmed_results=[],

        evidence=[
            (
                "Reporter失败前保留的压缩执行轨迹：\n"
                f"{trace_tail}"
            )
        ],

        errors=[
            (
                "StepReport结构化输出失败："
                f"{error[:2000]}"
            )
        ],

        unresolved_items=list(
            current_step.success_criteria
        ),

        next_action=(
            "后续节点可以继续运行，"
            "但不得仅依据Reporter失败"
            "假设底层Step已经成功或失败。"
        ),

        request_replan=False,

        replan_reason=None,
    )


def _report_matches_step(
    report: StepReport,
    current_step: PlanStep,
) -> bool:
    """检查报告是否对应当前Step。"""

    if (
        report.step_id
        != current_step.step_id
    ):
        return False

    reported_criteria = [
        item.criterion.strip()

        for item
        in report.criterion_results
    ]

    expected_criteria = [
        criterion.strip()

        for criterion
        in current_step.success_criteria
    ]

    return (
        reported_criteria
        == expected_criteria
        and (
            report.status != "COMPLETED"
            or all(item.status == "MET" for item in report.criterion_results)
        )
    )


def _reporter_skill_task(review_packet: StepReviewPacket) -> dict[str, Any]:
    """Expose bounded outcome clues, never raw Worker messages, to selection.

    This is discovery context, not a replacement for the independent review
    packet. Claims and excerpts remain untrusted data, not instructions.
    """
    return {
        "task_contract": {k: v for k, v in review_packet.task_contract.model_dump(mode="json").items()
                          if k in {"user_request", "plan_objective", "step_assignment", "success_criteria"}},
        "recent_attempt_outcomes": [{
            "finish_reason": attempt.finish_reason,
            "has_errors": any(row.get("status") == "ERROR" for row in attempt.tool_audit),
            "has_artifacts": bool(attempt.resolved_artifacts),
            "has_unresolved_items": bool(attempt.unresolved_items),
        } for attempt in review_packet.attempts[-2:]],
    }


async def run_step_reporter(
    simple_model,
    *,
    current_step: PlanStep,
    review_packet: StepReviewPacket,
    max_model_rounds: int,
    model_output_max_tokens: int,
    skill_catalog=None,
    skill_mode=None,
    skill_fixed_ids=(),
    saved_skill_snapshot=None,
) -> StepReportRunResult:
    """Review one bounded packet in a fresh context with registered material reads.

    这个函数：

    1. 不调用业务工具；
    2. 不修改原Plan；
    3. 不生成最终用户回答；
    4. Schema错误最多额外执行三次结构修复；
    5. 每一次真实模型请求都会计入预算；
    6. 只读取StepReviewPacket，不读取Worker原始消息；
    7. Packet异常超限时执行最后一道确定性长度保护；
    8. 最终失败时返回Python兜底报告。
    """

    step_execution_trace = review_packet.model_dump(
        mode="json",
    )
    from workers.evidence_refs import registry as evidence_registry, display as evidence_display, eligible_evidence
    reporter_evidence_refs = evidence_registry({'review_packet': step_execution_trace})
    completed_evidence_ids = eligible_evidence({'review_packet': step_execution_trace})
    for attempt in review_packet.attempts:
        for artifact in attempt.resolved_artifacts:
            if artifact.review_ref and artifact.review_ref not in reporter_evidence_refs:
                reporter_evidence_refs[artifact.review_ref] = f"A{sum(v.startswith('A') for v in reporter_evidence_refs.values()) + 1}"
    valid_reporter_refs = {raw: ref for raw, ref in reporter_evidence_refs.items()
                           if raw in completed_evidence_ids or ref.startswith("A")}
    step_execution_trace = evidence_display(step_execution_trace, reporter_evidence_refs)
    stop_reason = (
        review_packet.attempts[-1].stop_reason
        if review_packet.attempts
        else "Step review requested."
    )
    remaining_budget = review_packet.remaining_budget

    if (
        isinstance(
            max_model_rounds,
            bool,
        )
        or not isinstance(
            max_model_rounds,
            int,
        )
    ):
        raise TypeError(
            "max_model_rounds必须是整数。"
        )

    if max_model_rounds < 1:
        raise ValueError(
            "max_model_rounds不能小于1。"
        )

    if (
        isinstance(
            model_output_max_tokens,
            bool,
        )
        or not isinstance(
            model_output_max_tokens,
            int,
        )
    ):
        raise TypeError(
            "model_output_max_tokens必须是整数。"
        )

    if model_output_max_tokens < 512:
        raise ValueError(
            "model_output_max_tokens不能小于512。"
        )

    allowed_model_rounds = max_model_rounds

    schema_text = json.dumps(
        ReferencedStepReport.model_json_schema(),

        ensure_ascii=False,

        indent=2,
    )

    trace_placeholder = (
        "[STEP_REVIEW_PACKET_PLACEHOLDER]"
    )

    from skill_runtime import prepare_skills, skill_prompt
    from prompt_loader import split_prompt
    snapshot = await prepare_skills(
        simple_model, role="step_reporter",
        task=_reporter_skill_task(review_packet),
        catalog=skill_catalog, topics=[*current_step.skill_topics, *(["appworld"] if any(e.tool_name in {"appworld_discover", "appworld_execute", "appworld_verify"} for a in review_packet.attempts for e in a.resolved_evidence) else [])],
        mode=skill_mode, fixed_ids=skill_fixed_ids, saved=saved_skill_snapshot,
        allow_model=max_model_rounds > MAX_STEP_REPORT_RETRIES + 1,
    )
    preparation_calls = snapshot.model_calls if saved_skill_snapshot is None else 0
    allowed_model_rounds = min(allowed_model_rounds, max_model_rounds - preparation_calls)
    selected_methods = skill_prompt(snapshot)

    prompt_with_placeholder = render_prompt(
        "reporters/step_report",

        review_packet=(
            trace_placeholder
        ),
    )

    structured_output_suffix = (
        "\n\n"
        + render_prompt(
            "reporters/step_report_output",
            schema=schema_text,
        )
    )

    # 演示项目的简单启发式预算：
    #
    # CLOUD_LLM_MAX_TOKENS=5000时：
    #
    # total_budget_tokens = 15000
    # output_reserve_tokens = 5000
    #
    # 其余空间再扣除固定Prompt、Schema和安全余量，
    # 剩余部分才允许注入Step执行轨迹。
    total_budget_tokens = max(
        (
            model_output_max_tokens
            * STEP_REPORT_TOTAL_BUDGET_MULTIPLIER
        ),

        (
            model_output_max_tokens
            + STEP_REPORT_MIN_TRACE_TOKENS
        ),
    )

    output_reserve_tokens = max(
        512,

        (
            total_budget_tokens
            // STEP_REPORT_OUTPUT_RESERVE_DIVISOR
        ),
    )

    fixed_prompt_tokens = _estimate_tokens(
        prompt_with_placeholder
        + structured_output_suffix
        + selected_methods
    )

    trace_budget_tokens = max(
        STEP_REPORT_MIN_TRACE_TOKENS,

        (
            total_budget_tokens
            - output_reserve_tokens
            - fixed_prompt_tokens
            - STEP_REPORT_FIXED_SAFETY_TOKENS
        ),
    )

    (
        compacted_trace,
        trace_was_compacted,
        original_trace_tokens,
        compacted_trace_tokens,
    ) = _compact_step_execution_trace(
        step_execution_trace,

        max_tokens=(
            trace_budget_tokens
        ),
    )

    prompt = render_prompt(
        "reporters/step_report",

        review_packet=(
            _to_prompt_text(
                compacted_trace,

                "没有可用StepReviewPacket。",
            )
        ),
    )

    structured_model = (
        simple_model
        .with_structured_output(
            ReferencedStepReport,

            # 与Hard结构化节点保持一致。
            # Thinking模式下不依赖强制Tool Choice。
            method="json_mode",

            include_raw=True,
        )
    )

    messages: list[
        dict[
            str,
            str,
        ]
    ] = [
        {
            "role": "system",

            "content": (
                prompt
                + structured_output_suffix
            ),
        }
    ]

    fixed_prompt, packet_context = split_prompt(prompt)
    messages = [{"role": "system", "content": fixed_prompt + structured_output_suffix + "\n\n" + selected_methods},
                {"role": "user", "content": packet_context}]
    # The contract is never compressed with evidence; validation uses these exact criteria.
    messages.append({"role": "user", "content": "原始验收契约（逐条原样审核，不得自行拆分或替换）：" + review_packet.task_contract.model_dump_json()})
    # Publication identities must survive packet compaction; approval uses these exact refs.
    artifact_manifest = [{"review_ref": reporter_evidence_refs.get(a.review_ref, a.review_ref), "description": a.description[:300],
                          "size_bytes": a.size_bytes, "sha256": a.sha256}
                         for attempt in review_packet.attempts for a in attempt.resolved_artifacts]
    if artifact_manifest:
        messages.append({"role": "user", "content": "候选文件索引（存在不等于内容已验证）：" + json.dumps(artifact_manifest, ensure_ascii=False)})
    from reporting.materials import ReviewWithReads, build_reader
    _, material_refs = build_reader(review_packet)
    material_refs = evidence_display(material_refs, reporter_evidence_refs)
    if material_refs and hasattr(simple_model, "bind_tools") and allowed_model_rounds > 1:
        structured_model = ReviewWithReads(simple_model, structured_model, review_packet, allowed_model_rounds)
        messages.append({"role": "user", "content": "可按需调用read_review_material读取登记材料，或直接提交StepReport。最后一轮只提交报告。材料引用：" + json.dumps(material_refs, ensure_ascii=False)})
    material_read_pending = False
    model_rounds_used = preparation_calls

    last_error = (
        "未知StepReport错误。"
    )

    with trace_span(
        (
            "step_report."
            f"step_{current_step.step_id}"
        ),

        kind="chain",

        input_value={
            "step": (
                current_step
            ),

            "stop_reason": (
                stop_reason
            ),

            "remaining_budget": (
                remaining_budget
            ),

            "requested_model_rounds": (
                max_model_rounds
            ),

            "allowed_model_rounds": (
                allowed_model_rounds
            ),

            "model_output_max_tokens": (
                model_output_max_tokens
            ),

            "review_contract_preserved": True,
            "review_final_submission_reserve": min(2, allowed_model_rounds),
            "reporter_total_budget_tokens": (
                total_budget_tokens
            ),

            "reporter_output_reserve_tokens": (
                output_reserve_tokens
            ),

            "reporter_fixed_prompt_tokens": (
                fixed_prompt_tokens
            ),

            "reporter_trace_budget_tokens": (
                trace_budget_tokens
            ),

            "trace_was_compacted": (
                trace_was_compacted
            ),

            "original_trace_estimated_tokens": (
                original_trace_tokens
            ),

            "compacted_trace_estimated_tokens": (
                compacted_trace_tokens
            ),
        },

        attributes={
            "step_report.requested_model_rounds": (
                max_model_rounds
            ),
            "evidence.references_json": json.dumps({ref: raw for raw, ref in reporter_evidence_refs.items()}),
            "review.criteria_json": json.dumps(review_packet.task_contract.criterion_registry, ensure_ascii=False),

            "step_report.allowed_model_rounds": (
                allowed_model_rounds
            ),

            "step_report.output_max_tokens": (
                model_output_max_tokens
            ),

            "step_report.trace_budget_tokens": (
                trace_budget_tokens
            ),

            "step_report.trace_compacted": (
                trace_was_compacted
            ),
        },
    ) as span:

        for attempt_index in range(
            allowed_model_rounds
        ):
            if attempt_index > 0 and not material_read_pending:
                messages.append(
                    {
                        "role": "user",

                        "content": schema_repair_feedback(
                            schema_name="StepReport",
                            schema=ReferencedStepReport.model_json_schema(),
                            error_text=last_error[:1000],
                            instruction=(
                                "保留审核判断，只修正报告表单；criterion_results必须按C编号覆盖全部标准，"
                                "证据只引用已登记E/A编号，不要重新执行Worker业务。"
                            ),
                        ),
                    }
                )

            material_read_pending = False

            # 只要真实发起请求就计入预算。
            #
            # 包括：
            # - 正常返回；
            # - LengthFinishReasonError；
            # - 超时；
            # - 服务异常；
            # - 返回后解析失败。
            model_rounds_used += 1

            try:
                response = await (
                    structured_model
                    .ainvoke(
                        messages
                    )
                )

            except Exception as error:
                last_error = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                if (
                    attempt_index + 1 < allowed_model_rounds
                    and is_schema_repairable_error(error)
                ):
                    continue

                break

            if not isinstance(
                response,
                Mapping,
            ):
                last_error = (
                    "结构化模型没有返回Mapping。"
                )

                continue

            if response.get("material_read"):
                material_read_pending = True
                continue

            parsing_error = response.get(
                "parsing_error"
            )

            if parsing_error is not None:
                last_error = str(
                    parsing_error
                )

                continue

            try:
                report = (
                    normalize_report(response.get("parsed"), current_step.success_criteria, valid_reporter_refs)
                )

            except Exception as error:
                last_error = str(
                    error
                )

                continue

            if not _report_matches_step(
                report,
                current_step,
            ):
                last_error = (
                    "StepReport必须匹配当前step_id并按原顺序审核全部成功标准；"
                    "只有每项criterion_results.status都是MET才允许COMPLETED。"
                )

                continue

            set_span_attributes(
                span,

                **{
                    "step_report.success": True,

                    "step_report.status": (
                        report.status
                    ),

                    "step_report.model_rounds_used": (
                        model_rounds_used
                    ),

                    "step_report.validation_retry_count": (
                        attempt_index
                    ),

                    "step_report.used_fallback": False,
                },
            )

            set_span_output(
                span,

                {
                    "status": "success",

                    "report": (
                        report
                    ),

                    "model_rounds_used": (
                        model_rounds_used
                    ),

                    "validation_retry_count": (
                        attempt_index
                    ),

                    "allowed_model_rounds": (
                        allowed_model_rounds
                    ),

                    "trace_was_compacted": (
                        trace_was_compacted
                    ),
                },
            )

            return StepReportRunResult(
                skill_snapshot=snapshot.model_dump(mode="json"),
                report=(
                    report
                ),

                model_rounds_used=(
                    model_rounds_used
                ),

                used_fallback=False,
            )

        if (
            model_rounds_used
            >= allowed_model_rounds
        ):
            last_error += (
                "\n"
                "Step Reporter已经达到本次动态模型预算："
                f"{allowed_model_rounds}轮。"
            )

        fallback_report = (
            _build_report_fallback(
                current_step=(
                    current_step
                ),

                stop_reason=(
                    stop_reason
                ),

                error=(
                    last_error
                ),

                step_execution_trace=(
                    compacted_trace
                ),
            )
        )

        logger.warning(
            "StepReport生成失败，"
            "已使用安全兜底 | "
            "step_id=%s | rounds=%s/%s | "
            "trace_compacted=%s | error=%s",

            current_step.step_id,

            model_rounds_used,

            allowed_model_rounds,

            trace_was_compacted,

            last_error,
        )

        set_span_attributes(
            span,

            **{
                "step_report.success": False,

                "step_report.model_rounds_used": (
                    model_rounds_used
                ),

                "step_report.used_fallback": True,

                "step_report.budget_exhausted": (
                    model_rounds_used
                    >= allowed_model_rounds
                ),
            },
        )

        set_span_output(
            span,

            {
                "status": "fallback",

                "error": (
                    last_error
                ),

                "report": (
                    fallback_report
                ),

                "model_rounds_used": (
                    model_rounds_used
                ),

                "allowed_model_rounds": (
                    allowed_model_rounds
                ),

                "budget_exhausted": (
                    model_rounds_used
                    >= allowed_model_rounds
                ),

                "trace_was_compacted": (
                    trace_was_compacted
                ),
            },
        )

        return StepReportRunResult(
            skill_snapshot=snapshot.model_dump(mode="json"),
            report=(
                fallback_report
            ),

            model_rounds_used=(
                model_rounds_used
            ),

            used_fallback=True,
        )
