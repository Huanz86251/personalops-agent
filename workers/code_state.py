"""State and bounded prompt context shared by CODE-specific Deep Agents."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain.agents.middleware import (
    AgentMiddleware,
    hook_config,
)
from typing_extensions import NotRequired

from workers.progress import WorkerProgressState
from workers.runtime_events import context_event
from workers.code_review_models import CodeReviewLoopState


CodeAgentRole = Literal["WORKER", "REVIEWER"]


class CodeAgentState(WorkerProgressState):
    """Checkpoint fields exchanged by the Code Worker and Code Reviewer."""

    code_task: NotRequired[dict[str, Any]]
    code_prompt_parts: NotRequired[dict[str, str]]
    code_candidate: NotRequired[dict[str, Any]]
    code_review_loop: NotRequired[dict[str, Any]]
    code_repair_instruction: NotRequired[dict[str, Any]]
    final_worker_repair_request: NotRequired[dict[str, Any]]
    code_tool_audit: NotRequired[list[dict[str, Any]]]
    code_worker_submission: NotRequired[dict[str, Any]]
    code_worker_repair_response: NotRequired[dict[str, Any]]
    code_continuation_submission: NotRequired[dict[str, Any]]
    code_publication_context: NotRequired[dict[str, Any]]
    code_artifact_manifest: NotRequired[dict[str, Any]]
    code_publication_receipt: NotRequired[dict[str, Any]]
    code_review_report: NotRequired[dict[str, Any]]
    code_agent_finished: NotRequired[bool]


def _json_block(title: str, value: Any) -> str | None:
    if value is None or value == "" or value == {} or value == []:
        return None
    return (
        f"[{title}]\n"
        + json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    )


class CodeRuntimeContextMiddleware(AgentMiddleware[CodeAgentState]):
    """Inject only the persisted contract needed by one CODE role."""

    state_schema = CodeAgentState

    def __init__(self, role: CodeAgentRole) -> None:
        self.role = role

    def before_model(self, state: CodeAgentState, runtime):
        if state.get("code_agent_finished"):
            return None
        raw_loop = state.get("code_review_loop")
        loop_context = raw_loop
        role_guidance = None
        current_repair_instruction = state.get("code_repair_instruction")
        if raw_loop:
            loop = CodeReviewLoopState.model_validate(raw_loop)
            current_repair_instruction = (
                loop.pending_instruction.model_dump(mode="json")
                if loop.pending_instruction is not None
                else None
            )
            loop_context = loop.model_dump(mode="json")
            loop_context.pop("pending_scheduler_directive", None)
            loop_context.pop("scheduler_continuations", None)
            for key in ("exchanges", "worker_checkpoint_id", "reviewer_checkpoint_id", "candidate", "pending_instruction"):
                loop_context.pop(key, None)
            directive = loop.pending_scheduler_directive
            response = None
            if directive is None and loop.scheduler_continuations:
                latest = loop.scheduler_continuations[-1]
                if latest.directive.scheduler_epoch == loop.scheduler_epoch:
                    directive = latest.directive
                    response = latest.submission
            if directive is not None:
                role_guidance = {
                    "scheduler_epoch": directive.scheduler_epoch,
                    "reason": directive.reason,
                    "instruction": (
                        directive.worker_instruction
                        if self.role == "WORKER"
                        else directive.reviewer_instruction
                    ),
                }
                if self.role == "REVIEWER" and response is not None:
                    role_guidance["worker_submission"] = response.model_dump(
                        mode="json"
                    )
        blocks = [
            _json_block("代码任务", state.get("code_task")),
            _json_block(
                "Final Reviewer事实型返修单（只说明验收缺口，不代表接口或执行步骤）",
                state.get("final_worker_repair_request"),
            ),
            _json_block("当前候选", state.get("code_candidate")),
            _json_block("审核进度", loop_context),
            _json_block("继续执行要求", role_guidance),
            _json_block(
                "返修要求",
                current_repair_instruction,
            ),
        ]
        if self.role == "REVIEWER":
            blocks.extend(
                [
                    _json_block("本代码流程工具记录（可能含前轮审核调用）", state.get("code_tool_audit")),
                    _json_block(
                        "最新代码提交",
                        state.get("code_worker_submission"),
                    ),
                    _json_block(
                        "发布清单",
                        state.get("code_artifact_manifest"),
                    ),
                    _json_block(
                        "发布回执",
                        state.get("code_publication_receipt"),
                    ),
                ]
            )
        previous = state.get("code_prompt_parts", {})
        current = {block.split("\n", 1)[0]: block for block in blocks if block}
        changed = [text for key, text in current.items() if previous.get(key) != text]
        changed += [key + "\nnull" for key in previous if key not in current]
        if not changed:
            return None
        update = context_event(state, "\n\n".join(changed), "code_context_fingerprint") or {}
        update["code_prompt_parts"] = current
        return update


class CodeAgentCompletionMiddleware(AgentMiddleware[CodeAgentState]):
    """End a role turn immediately after its structured handoff is stored."""

    state_schema = CodeAgentState

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: CodeAgentState,
        runtime,
    ) -> dict[str, Any] | None:
        if state.get("code_agent_finished"):
            return {"jump_to": "end"}
        return None


__all__ = [
    "CodeAgentCompletionMiddleware",
    "CodeAgentRole",
    "CodeAgentState",
    "CodeRuntimeContextMiddleware",
]
