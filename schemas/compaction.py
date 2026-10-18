"""Strict output contracts; optional semantics still have required API fields."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    user_requirements: list[str] = Field(description="用户明确提出且仍有效的目标、约束和偏好；无内容时为空数组。")
    decisions: list[str] = Field(description="已确认的最新决定，不把助手建议当作用户同意；无内容时为空数组。")
    assistant_results: list[str] = Field(description="用户收到的助手结果，区分助手自报与有证据的验证；无内容时为空数组。")
    pending_items: list[str] = Field(description="尚未完成的工作、阻塞和待确认问题；删除已完成或取消项，无内容时为空数组。")
    important_references: list[str] = Field(description="必要的准确文件路径、标识、数字及错误信息；无内容时为空数组。")


class SummaryCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1
    covered_turns: int
    summary: ConversationSummary


class WorkerExecutionSummary(BaseModel):
    """Execution evidence, not a new plan, acceptance decision or final answer."""
    model_config = ConfigDict(extra="forbid", strict=True)
    observed_results: list[str] = Field(description="工具或执行材料实际观察到的结果。区分成功、助手自报与尚未验证；无内容时为空数组。")
    errors_and_failures: list[str] = Field(description="影响继续执行的错误、失败尝试和原因；保留尚未解决的失败，不将失败写成成功；无内容时为空数组。")
    pending_work: list[str] = Field(description="尚未完成的工作、缺少的信息与验证；不能把计划写成已完成，无内容时为空数组。")
    files_and_artifacts: list[str] = Field(description="需要继续使用的准确文件路径、产物和相关关键数值；无内容时为空数组。")
    evidence_refs: list[str] = Field(description="只保留仍与当前执行相关且输入真实存在的证据引用ID，不发明新ID；无关引用可省略，没有必要引用时为空数组。")
    next_steps: list[str] = Field(description="基于现有结果可继续的动作建议，不改变用户授权或任务要求；无内容时为空数组。")
