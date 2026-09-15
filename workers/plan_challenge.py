"""Structured Worker challenge to a possibly incorrect planning contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlanChallenge(BaseModel):
    """Evidence-first objection that must be reviewed before replanning."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reason: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "先说明为什么当前Step或范围关系与用户原话或真实只读结果冲突；"
            "API不熟、单次报错或执行困难不构成计划异议。"
        ),
    )
    original_requirement: str = Field(
        min_length=1,
        max_length=800,
        description="支持异议的用户原始要求，保留关键范围词和条件归属。",
    )
    conflicting_plan_text: str = Field(
        min_length=1,
        max_length=800,
        description="当前ScopeContract、target_selection或Step中与原要求冲突的具体内容。",
    )
    evidence_tool_call_ids: list[str] = Field(
        default_factory=list,
        description="可选的真实只读证据编号；若冲突仅凭用户原话即可确认可为空。",
    )
    requested_revision: str = Field(
        min_length=1,
        max_length=800,
        description=(
            "只描述应重新确认的对象、条件归属、集合关系或读写效果，"
            "不猜API，不替Scheduler生成新计划。"
        ),
    )

    @field_validator("evidence_tool_call_ids")
    @classmethod
    def normalize_evidence_ids(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            item = str(value).strip()
            if item and item not in result:
                result.append(item)
        return result


__all__ = ["PlanChallenge"]
