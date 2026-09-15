"""General submits a bounded handoff for deterministic review routing."""

from datetime import datetime, timezone
from typing import Literal

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic.json_schema import SkipJsonSchema
from api_handoff import ApiHandoffList, api_field
from handoff_knowledge import HandoffKnowledge, knowledge_field
from pydantic import BaseModel, Field
from prompt_loader import load_prompt

from artifact_models import WorkerArtifactCandidate
from workers.progress import WorkerProgressState
from workers.submission import (
    WorkerSubmission,
    WorkerSubmissionRecord,
    resolve_artifact_candidates,
    resolve_tool_evidence,
)
from workers.plan_challenge import PlanChallenge
from middlewares import DynamicExecutionBudgetMiddleware


class GeneralBudgetMiddleware(DynamicExecutionBudgetMiddleware):
    """Reserve the last model request for reporting, inside the existing limit."""

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if (
            state.get("worker_finalize_requested")
            and state.get("worker_finalize_reason") == "SCHEMA_REPAIR"
        ):
            repair_used = int(state.get("worker_schema_repair_model_calls_used", 0) or 0)
            repair_limit = max(
                int(
                    state.get(
                        "worker_schema_repair_model_run_limit",
                        self.schema_repair_max_rounds,
                    )
                    or self.schema_repair_max_rounds
                ),
                1,
            )
            return None if repair_used < repair_limit else {"jump_to": "end"}
        finalization_limit = max(
            int(state.get("worker_finalization_model_run_limit", self.finalization_model_rounds) or self.finalization_model_rounds),
            1,
        )
        if state.get("worker_finalize_requested") and int(state.get("worker_finalization_model_calls_used", 0) or 0) >= finalization_limit:
            return {"jump_to": "end"}
        limit = state.get("executor_model_run_limit")
        if limit is None:
            return None
        used = sum(int(state.get(k, 0) or 0) for k in (
            "executor_model_calls_used", "skill_preparation_calls_used", "worker_compaction_calls_used"))
        if used >= limit:
            return {"jump_to": "end"}
        tools_used = int(state.get("executor_tool_calls_used", 0) or 0)
        tools_limit = state.get("executor_tool_run_limit")
        if used >= max(0, limit - 1) or (tools_limit is not None and tools_used >= tools_limit):
            return {"worker_finalize_requested": True,
                    "worker_finalize_reason": "执行额度已到收尾边界，剩余调用仅用于总结"}
        return None

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        if not state.get("worker_finalize_requested"):
            update = super().after_model(state, runtime)
            if update and update.get("messages"):
                last = update["messages"][-1]
                if not getattr(last, "tool_calls", []):
                    update.update(worker_finalize_requested=True,
                                  worker_finalize_reason="工具调用被预算限制，停止执行并总结",
                                  jump_to="model")
            return update
        # Submission is a control operation, not another business tool action.
        from langchain.messages import RemoveMessage
        from langgraph.graph.message import REMOVE_ALL_MESSAGES
        messages = list(state.get("messages", []))
        last = messages[-1]
        calls = getattr(last, "tool_calls", [])
        allowed = [c for c in calls if c.get("name") == GENERAL_REPORT_NAME][:1]
        update = {
            "worker_finalization_model_calls_used": int(state.get("worker_finalization_model_calls_used", 0)) + 1,
            "worker_finalization_tool_calls_used": int(state.get("worker_finalization_tool_calls_used", 0)) + len(allowed),
        }
        if state.get("worker_finalize_reason") == "SCHEMA_REPAIR":
            update["worker_schema_repair_model_calls_used"] = (
                int(state.get("worker_schema_repair_model_calls_used", 0) or 0) + 1
            )
        else:
            update["executor_model_calls_used"] = int(state.get("executor_model_calls_used", 0)) + 1
        if calls != allowed:
            update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[:-1],
                                  last.model_copy(update={"tool_calls": allowed})]
        return update

GENERAL_REPORT_NAME = "report_general_result"


class GeneralFile(BaseModel):
    path: str = Field(
        pattern=r"^/(artifacts|downloads)/.+",
        description="Existing checkpoint artifact or private /downloads/<candidate_id>; shared publication requires review.",
    )
    description: str = Field(
        min_length=1, max_length=600, description="What this file contains."
    )
    output_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_-]*$",
        max_length=64,
        description="Exact output slot assigned by Scheduler, if any.",
    )


from workers.submission import WorkerCriterionClaim


class GeneralResult(BaseModel):
    forced_finalization: SkipJsonSchema[bool] = Field(default=False, description="Harness-only forced finalization marker.")
    criterion_claims: list[WorkerCriterionClaim] = Field(default_factory=list, description="For each current criterion, give your claim and its evidence. Claims are not independent approval.")
    handoff_apis: ApiHandoffList = api_field()
    handoff_knowledge: SkipJsonSchema[list[HandoffKnowledge]] = knowledge_field()
    plan_challenge: PlanChallenge | None = Field(
        default=None,
        description=(
            "当前Step或ScopeContract实质改变了用户要求的对象、条件归属、集合关系"
            "或读写效果时填写；不要把接口不熟、调用失败或执行困难写成计划异议。"
        ),
    )

    status: Literal["COMPLETED", "PARTIAL", "BLOCKED", "FAILED"] = Field(
        description="自查后的状态。正常COMPLETED且无文件直接形成StepReport；未完成、预算收尾或有文件时由Harness启动独立Reviewer。"
    )
    summary: str = Field(
        min_length=1,
        max_length=4000,
        description="Actual completed actions and observed results. Distinguish not attempted from attempted and failed; do not include passwords or tokens.",
    )
    unresolved_items: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="For each unfinished item: not attempted or actual failure, observed blocker, missing prerequisite and next action. Execution-round exhaustion is not an API timeout or account-credit failure.",
    )
    evidence_tool_call_ids: list[str] = Field(
        default_factory=list,
        description="Cite every relevant, non-redundant actual tool result needed for coverage; omit for tasks requiring no tools.",
    )
    files: list[GeneralFile] = Field(
        default_factory=list,
        max_length=12,
        description="Generated files needed by subsequent steps or requested as deliverables; omit for a text-only result.",
    )


@tool(description=load_prompt("workers/general_report_tool"))
def report_general_result(result: GeneralResult, runtime: ToolRuntime) -> Command:
    """Finish a simple General step with your self-checked results. This ends execution; Harness decides whether independent review is needed. Report incomplete work honestly; include real artifact candidates if later workers need generated files."""
    state = runtime.state
    if result.plan_challenge is not None and int(
        state.get("worker_plan_challenges_remaining", 0) or 0
    ) <= 0:
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    "Plan challenge rejected by Harness: the global challenge budget is exhausted. "
                    "Continue within the current Step when possible; otherwise report the real unresolved "
                    "result with plan_challenge=null. No Reviewer or Scheduler was called."
                ),
                tool_call_id=runtime.tool_call_id or GENERAL_REPORT_NAME,
                name=GENERAL_REPORT_NAME,
                status="error",
            )],
        })
    from reporting.criteria import (
        criterion_claim_repair_feedback,
        validate_worker_claim_coverage,
    )
    try:
        validate_worker_claim_coverage(state, result.criterion_claims)
    except ValueError as error:
        return Command(update={
            "worker_finalize_requested": True,
            "worker_finalize_reason": "SCHEMA_REPAIR",
            "messages": [ToolMessage(
                content=criterion_claim_repair_feedback(state, error),
                tool_call_id=runtime.tool_call_id or GENERAL_REPORT_NAME,
                name=GENERAL_REPORT_NAME,
                status="error",
            )],
        })
    result = result.model_copy(update={
        'forced_finalization': bool(
            state.get('worker_finalize_requested')
            and state.get('worker_finalize_reason') != 'SCHEMA_REPAIR'
        )
    })
    ids = list(result.evidence_tool_call_ids)
    ids.extend(ref for claim in result.criterion_claims for ref in claim.evidence_tool_call_ids)
    if result.plan_challenge is not None:
        ids.extend(result.plan_challenge.evidence_tool_call_ids)
    candidates = []
    downloads = {r["candidate_id"]: r for r in state.get("worker_downloaded_artifacts", [])}
    for index, file in enumerate(result.files):
        if file.path.startswith("/downloads/"):
            candidate_id = file.path.removeprefix("/downloads/")
            download = downloads.get(candidate_id)
            if not download:
                raise ValueError("File is not registered to this Worker")
            candidates.append(WorkerArtifactCandidate(
                candidate_id=candidate_id, kind="DOWNLOADED_FILE", description=file.description,
                output_id=file.output_id, evidence_tool_call_ids=[download["tool_call_id"]],
            ))
        else:
            candidates.append(WorkerArtifactCandidate(
                candidate_id=f"general-file-{index}", kind="WORKSPACE_FILE", **file.model_dump(),
            ))
    artifacts = resolve_artifact_candidates(state, candidates)
    ids = list(dict.fromkeys([*ids, *(call_id for candidate in candidates for call_id in candidate.evidence_tool_call_ids)]))
    evidence = resolve_tool_evidence(list(state.get("worker_archived_messages", [])) + list(state.get("messages", [])), ids)
    # Reuse the durable artifact/evidence envelope, never the review protocol.
    record = WorkerSubmissionRecord(
        submitted_at=datetime.now(timezone.utc),
        worker_id=state.get("worker_id"),
        event_id=state.get("event_id"),
        step_id=state.get("step_id"),
        total_tool_calls=int(state.get("worker_total_tool_calls", 0)),
        submission=WorkerSubmission(
            plan_challenge=result.plan_challenge,
            criterion_claims=result.criterion_claims,
            summary=result.summary[:2000],
            final_conclusion=result.summary[:1600],
            unresolved_items=result.unresolved_items,
            handoff_knowledge=result.handoff_knowledge,
            handoff_apis=result.handoff_apis,
        ),
        resolved_evidence=evidence,
        resolved_artifacts=artifacts,
    )
    return Command(
        update={
            "general_result": result.model_dump(mode="json"),
            "worker_submission": record.model_dump(mode="json"),
            "messages": [
                ToolMessage(
                    content=("General self-report recorded; files remain private pending independent Reporter review."
                             if artifacts else "General self-report recorded. No independent review was performed."),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class GeneralCompletionMiddleware(AgentMiddleware[WorkerProgressState]):
    """Keep checkpoints and append a bounded final reporting instruction."""

    state_schema = WorkerProgressState

    def before_agent(self, state, runtime):
        messages = state.get("messages") or []
        if messages and getattr(messages[-1], "type", "") == "human":
            return {
                "general_result": {},
                "worker_submission": {},
                "worker_finalize_requested": False,
                "worker_schema_repair_model_calls_used": 0,
            }
        return None

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if (
            state.get("general_result")
            or state.get("worker_terminal_action") in {"CANCEL", "REPLACE", "ACCEPT"}
        ):
            return {"jump_to": "end"}
        if state.get("worker_finalize_requested"):
            from workers.runtime_events import context_event
            return context_event(state,
                "执行已停止。仅根据已有任务、工具结果和错误提交 report_general_result；禁止继续查询或修改。"
                "总结实际完成项；每个未完成项说明未尝试或真实失败、当前缺项及下一步。"
                "执行轮次耗尽不代表接口超时或账户额度不足；没有证据不得声称成功，不复制密码或令牌。"
                "现在提交报告。",
                "worker_control_fingerprint")
        return None

    def _final_request(self, request):
        if not request.state.get("worker_finalize_requested"):
            return request
        tools = [t for t in request.tools if (getattr(t, "name", None) or
                 (t.get("name") if isinstance(t, dict) else None)) == GENERAL_REPORT_NAME]
        return request.override(tools=tools, tool_choice=GENERAL_REPORT_NAME)

    def wrap_model_call(self, request, handler):
        return handler(self._final_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._final_request(request))

    def after_model(self, state, runtime):
        messages = state.get("messages") or []
        calls = getattr(messages[-1], "tool_calls", []) if messages else []
        if calls:
            return {
                "worker_total_tool_calls": int(state.get("worker_total_tool_calls", 0))
                + len(calls)
            }
        # A natural answer is already a report. Never ask another model round
        # merely to reformat it. The scheduler marks unstructured exits partial.
        return None
