"""Small public contracts for a Worker's mid-execution planning objection."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkerPlanObjection(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    problem: str = Field(min_length=1, max_length=1200, description="当前步骤与用户目标的具体冲突；接口不熟或一次报错不算计划异议。")
    reason: str = Field(min_length=1, max_length=1200, description="为什么继续按当前步骤执行会偏离用户要求。")
    suggested_direction: str = Field(min_length=1, max_length=1200, description="建议重新确认或调整的方向；不编造API，不代写完整计划。")


class WorkerObjectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    action: Literal["CONTINUE", "GUIDE", "REPLAN", "FINISH"]
    reason: str = Field(min_length=1, max_length=1600)
    guidance: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def require_guidance(self):
        if self.action == "GUIDE" and not (self.guidance or "").strip():
            raise ValueError("GUIDE requires guidance")
        if self.action != "GUIDE":
            self.guidance = None
        return self
