"""One-shot Worker request for additional capability groups.

The tool does not select tools.  It records a grounded capability gap for the
ToolsetRouterMiddleware.  A second request in the same Worker attempt switches
the existing Worker finalization path on instead of opening another catalog.
"""

from __future__ import annotations

from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator

from prompt_loader import load_prompt
from reporting.criteria import worker_registry
from workers.evidence_refs import eligible_evidence, registry as evidence_registry
from workers.submission import resolve_tool_evidence
from capability_expansion_contract import (
    CAPABILITY_EXPANSION_EXHAUSTED,
    CAPABILITY_EXPANSION_REQUESTED,
    REQUEST_ADDITIONAL_CAPABILITIES_NAME,
)


def _compact(value: str, *, maximum: int) -> str:
    return " ".join(str(value or "").strip().split())[:maximum]


class CapabilityExpansionRequest(BaseModel):
    """Model-authored, capability-level explanation of a real blocker."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    blocked_objective: str = Field(
        min_length=1,
        max_length=600,
        description="当前Step中还无法完成的目标；只写业务目标，不写API或工具名。",
    )
    missing_capability: str = Field(
        min_length=1,
        max_length=300,
        description="缺少的能力类别；例如联网检索、读取主机文件或向用户交付文件。不得填写API、函数或工具组名称。",
    )
    why_current_capabilities_are_insufficient: str = Field(
        min_length=1,
        max_length=1000,
        description="依据已经观察到的事实，说明当前可见能力为什么无法继续；不得猜测不存在的API。",
    )
    evidence_refs: list[str] = Field(
        min_length=1,
        max_length=8,
        description=(
            "支持本次申请的引用。U1代表用户原始请求，S1代表当前Step；"
            "也可引用界面已显示的C1/C2验收条件或E1/E2真实工具结果。不得编造编号。"
        ),
    )

    @field_validator("blocked_objective", "missing_capability", "why_current_capabilities_are_insufficient")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = _compact(value, maximum=1000)
        if not normalized:
            raise ValueError("capability gap text cannot be empty")
        return normalized

    @field_validator("evidence_refs")
    @classmethod
    def normalize_refs(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            ref = str(value or "").strip().upper()
            if ref and ref not in normalized:
                normalized.append(ref)
        if not normalized:
            raise ValueError("at least one evidence reference is required")
        return normalized


def _resolve_context_refs(state: Any, refs: list[str]) -> list[dict[str, Any]]:
    evidence_refs = evidence_registry(state)
    inverse_evidence = {short.upper(): raw for raw, short in evidence_refs.items()}
    allowed_evidence = eligible_evidence(state)
    criteria = worker_registry(state)
    resolved: list[dict[str, Any]] = []

    for ref in refs:
        if ref == "U1":
            value = str(
                state.get("toolset_route_full_user_request")
                or state.get("host_original_user_request")
                or ""
            ).strip()
            if not value:
                raise ValueError("U1 is unavailable in the current Worker state")
            resolved.append({"ref": ref, "kind": "USER_REQUEST", "content": value[:4000]})
            continue
        if ref == "S1":
            value = str(
                state.get("host_current_step")
                or state.get("toolset_route_query")
                or ""
            ).strip()
            if not value:
                raise ValueError("S1 is unavailable in the current Worker state")
            resolved.append({"ref": ref, "kind": "CURRENT_STEP", "content": value[:4000]})
            continue
        if ref in criteria:
            resolved.append({"ref": ref, "kind": "SUCCESS_CRITERION", "content": str(criteria[ref])[:2000]})
            continue
        raw = inverse_evidence.get(ref)
        if raw is None or raw not in allowed_evidence:
            raise ValueError(f"unknown or ineligible evidence reference: {ref}")
        evidence = resolve_tool_evidence(
            list(state.get("worker_archived_messages", [])) + list(state.get("messages", [])),
            [raw],
        )[0]
        # The selector needs the observed result, not an API recommendation.
        resolved.append({
            "ref": ref,
            "kind": "TOOL_RESULT",
            "content": str(evidence.result)[:4000],
        })
    return resolved


@tool(
    REQUEST_ADDITIONAL_CAPABILITIES_NAME,
    description=load_prompt("tools/request_additional_capabilities"),
)
def request_additional_capabilities(
    request: CapabilityExpansionRequest,
    runtime: ToolRuntime,
) -> Command:
    """Record one grounded request; a second request forces factual closeout."""

    state = runtime.state
    prior = list(state.get("capability_expansion_requests", []) or [])
    if prior:
        return Command(update={
            "capability_expansion_requests": [
                *prior,
                {
                    "status": CAPABILITY_EXPANSION_EXHAUSTED,
                    "missing_capability": request.missing_capability,
                },
            ],
            "worker_finalize_requested": True,
            "worker_finalize_reason": (
                "能力补选额度已经使用；停止调用业务工具。只依据现有证据提交PARTIAL或FAILED报告，"
                "写明已完成内容、未完成条件和仍缺少的能力；不要填写、推荐或猜测任何API名称。"
            ),
            "messages": [ToolMessage(
                content=(
                    f"{CAPABILITY_EXPANSION_EXHAUSTED}：本Step已经补选过一次能力。"
                    "现在必须停止执行并提交真实的PARTIAL或FAILED交接；只描述缺少的能力，"
                    "不要填写、推荐或猜测API名称。"
                ),
                tool_call_id=runtime.tool_call_id or REQUEST_ADDITIONAL_CAPABILITIES_NAME,
                name=REQUEST_ADDITIONAL_CAPABILITIES_NAME,
                status="error",
            )],
        })

    try:
        resolved = _resolve_context_refs(state, request.evidence_refs)
    except (IndexError, TypeError, ValueError) as error:
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    f"能力补选申请被拒绝：{error}。请只使用U1、S1、当前C编号或已显示的E编号，"
                    "并继续使用现有能力；不要猜测API名称。"
                ),
                tool_call_id=runtime.tool_call_id or REQUEST_ADDITIONAL_CAPABILITIES_NAME,
                name=REQUEST_ADDITIONAL_CAPABILITIES_NAME,
                status="error",
            )],
        })

    record = {
        "status": CAPABILITY_EXPANSION_REQUESTED,
        "blocked_objective": request.blocked_objective,
        "missing_capability": request.missing_capability,
        "why_current_capabilities_are_insufficient": request.why_current_capabilities_are_insufficient,
        "evidence_refs": request.evidence_refs,
        "resolved_evidence": resolved,
    }
    return Command(update={
        "capability_expansion_requests": [record],
        "messages": [ToolMessage(
            content=(
                f"{CAPABILITY_EXPANSION_REQUESTED}：证据引用已由Harness核验。"
                "正在从完整能力目录补选最多三组能力；原有能力继续可用。"
            ),
            tool_call_id=runtime.tool_call_id or REQUEST_ADDITIONAL_CAPABILITIES_NAME,
            name=REQUEST_ADDITIONAL_CAPABILITIES_NAME,
        )],
    })


__all__ = [
    "CAPABILITY_EXPANSION_EXHAUSTED",
    "CAPABILITY_EXPANSION_REQUESTED",
    "CapabilityExpansionRequest",
    "REQUEST_ADDITIONAL_CAPABILITIES_NAME",
    "request_additional_capabilities",
]
