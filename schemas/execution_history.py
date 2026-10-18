"""Model-facing protocol for two-stage, step-scoped execution history."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ExecutionHistoryQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_ids: list[Annotated[int, Field(gt=0, strict=True)]] | None = Field(
        default=None,
        description="要查看的步骤编号列表，可填多个，例如[1,2,3]。第一次查询请填写步骤编号并将references留空。省略或填[]时列出当前任务全部步骤的调用清单。",
    )
    references: list[str] | None = Field(
        default=None,
        description="第一次查询请留空；拿到调用清单后原样复制短引用，例如['S1-2','S2-3']。返回命中引用的完整工具结果，可省略step_ids。部分引用无效时返回命中结果及未命中引用；全部未命中时返回调用清单。",
    )
