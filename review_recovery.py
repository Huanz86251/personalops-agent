"""Role-specific recovery proposals produced after independent General/Code review.

The first review remains a truth/acceptance judgment.  Only when that review
escalates does the Harness ask the same reviewer role for a small recovery
proposal.  The proposal is advice to the Scheduler, never a rewritten Plan.

Web is intentionally different: ``WebRecoveryProposal`` is part of the Web
Worker-owned StepReport schema, so no second recovery-model call is registered
for WEB here.
"""

from __future__ import annotations

from typing import Any, Literal
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from observability import set_span_attributes, set_span_output, trace_span
from prompt_loader import load_prompt
from schema_normalization import ReportItem, EvidenceRefText, coerce_text_sequence


class _RecoveryProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        min_length=1,
        max_length=5000,
        description="无法满足冻结合同的已观察原因，最多5000字符。",
    )
    preserved_results: list[ReportItem] = Field(
        default_factory=list,
        max_length=24,
        description="最多24项已验证结果，每项最多2000字符；后续计划不得重复或抹除。",
    )
    downgrade_recommended: bool = Field(
        description=(
            "True only when the Reviewer recommends changing a material user requirement. "
            "Choosing a different execution path while preserving the contract is not a downgrade."
        ),
    )
    downgrade_reason: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description="Why the original requirement cannot be retained, or null when no downgrade is recommended.",
    )
    proposed_degraded_outcome: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description="Exact weaker outcome proposed to the user, or null when no downgrade is recommended.",
    )
    user_visible_disclosure: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description="Plain disclosure of the lost requirement, or null when no downgrade is recommended.",
    )
    user_confirmation_request: str | None = Field(
        default=None,
        min_length=1,
        max_length=5000,
        description=(
            "A complete, user-facing confirmation question. It must explain the original requirement, "
            "the verified blocker, the exact proposed downgrade, and ask whether that downgrade is accepted."
        ),
    )

    @field_validator("preserved_results", mode="before")
    @classmethod
    def normalize_preserved_results(cls, value):
        return coerce_text_sequence(value)

    @model_validator(mode="after")
    def validate_downgrade_shape(self):
        downgrade_fields = (
            self.downgrade_reason,
            self.proposed_degraded_outcome,
            self.user_visible_disclosure,
            self.user_confirmation_request,
        )
        if self.downgrade_recommended and any(value is None for value in downgrade_fields):
            raise ValueError(
                "downgrade_recommended=true requires downgrade_reason, "
                "proposed_degraded_outcome, user_visible_disclosure, and "
                "user_confirmation_request."
            )
        if not self.downgrade_recommended and any(value is not None for value in downgrade_fields):
            raise ValueError(
                "No-downgrade proposals must leave all downgrade and user-confirmation fields null."
            )
        return self


class WebRecoveryProposal(_RecoveryProposal):
    proposal_kind: Literal["WEB"] = "WEB"
    affected_criterion_ids: list[EvidenceRefText] = Field(min_length=1, max_length=24)
    unavailable_or_uncertain_facts: list[ReportItem] = Field(min_length=1, max_length=24)
    attempted_approaches: list[ReportItem] = Field(default_factory=list, max_length=24)
    strongest_supported_claim: str = Field(
        min_length=1,
        max_length=5000,
        description="现有证据支持的最强表述，最多5000字符；不得发明新事实。",
    )

    @field_validator("affected_criterion_ids", "unavailable_or_uncertain_facts", "attempted_approaches", mode="before")
    @classmethod
    def normalize_web_lists(cls, value):
        return coerce_text_sequence(value)


class GeneralRecoveryProposal(_RecoveryProposal):
    proposal_kind: Literal["GENERAL"] = "GENERAL"
    affected_criterion_ids: list[EvidenceRefText] = Field(min_length=1, max_length=24)
    blocked_requirements: list[ReportItem] = Field(min_length=1, max_length=24)
    external_state_impact: str = Field(
        min_length=1,
        max_length=4000,
        description="State what would and would not change in external systems.",
    )
    preserved_safety_boundaries: list[ReportItem] = Field(default_factory=list, max_length=24)

    @field_validator("affected_criterion_ids", "blocked_requirements", "preserved_safety_boundaries", mode="before")
    @classmethod
    def normalize_general_lists(cls, value):
        return coerce_text_sequence(value)


class CodeRecoveryProposal(_RecoveryProposal):
    proposal_kind: Literal["CODE"] = "CODE"
    affected_requirement_ids: list[EvidenceRefText] = Field(min_length=1, max_length=24)
    candidate_revision: str = Field(min_length=1, max_length=200)
    verified_files: list[EvidenceRefText] = Field(default_factory=list, max_length=100)
    verified_requirements: list[EvidenceRefText] = Field(default_factory=list, max_length=40)
    unmet_requirements: list[ReportItem] = Field(min_length=1, max_length=24)
    retained_interfaces: list[ReportItem] = Field(default_factory=list, max_length=24)
    validation_required_after_change: list[ReportItem] = Field(min_length=1, max_length=24)

    @field_validator(
        "affected_requirement_ids", "verified_files", "verified_requirements",
        "unmet_requirements", "retained_interfaces", "validation_required_after_change",
        mode="before",
    )
    @classmethod
    def normalize_code_lists(cls, value):
        return coerce_text_sequence(value)


PROPOSAL_MODELS: dict[str, type[_RecoveryProposal]] = {
    "GENERAL": GeneralRecoveryProposal,
    "CODE": CodeRecoveryProposal,
}

PROPOSAL_PROMPTS = {
    "GENERAL": "reviewers/recovery_general",
    "CODE": "reviewers/recovery_code",
}


async def build_recovery_proposal(
    model,
    *,
    worker_kind: str,
    user_request: str,
    current_step: Any,
    review_output: Any,
    trigger_source: str,
) -> tuple[dict[str, Any] | None, int, str | None]:
    """Ask the reviewer role once for bounded recovery advice.

    Raw Worker messages and review materials are deliberately absent.  The
    reviewer already distilled them in ``review_output``; the Scheduler should
    receive that judgment, not a second copy of execution context.
    """

    kind = str(worker_kind or "").upper()
    schema = PROPOSAL_MODELS.get(kind)
    prompt_name = PROPOSAL_PROMPTS.get(kind)
    if schema is None or prompt_name is None:
        return None, 0, f"Unsupported recovery proposal role: {kind or 'UNKNOWN'}"

    payload = {
        "trigger_source": trigger_source,
        "user_request": user_request,
        "current_step": (
            current_step.model_dump(mode="json")
            if hasattr(current_step, "model_dump")
            else current_step
        ),
        "review_output": (
            review_output.model_dump(mode="json")
            if hasattr(review_output, "model_dump")
            else review_output
        ),
    }
    from structured_output_policy import (
        textual_schema_fallback,
        with_provider_structured_output,
    )
    structured = with_provider_structured_output(
        model,
        schema,
        include_raw=True,
        fallback_method="json_mode",
    )
    schema_guidance = textual_schema_fallback(model, schema)
    messages = [
        {
            "role": "system",
            "content": (
                load_prompt(prompt_name)
                + (("\n\n" + schema_guidance) if schema_guidance else "")
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ),
        },
    ]

    with trace_span(
        f"review_recovery.{kind.lower()}",
        kind="chain",
        input_value={
            "trigger_source": trigger_source,
            "worker_kind": kind,
            "step_id": getattr(current_step, "step_id", None),
        },
    ) as span:
        try:
            response = await structured.ainvoke(messages)
            if not isinstance(response, dict):
                raise ValueError("Recovery proposal model did not return a mapping")
            if response.get("parsing_error") is not None:
                raise ValueError(str(response["parsing_error"]))
            parsed = response.get("parsed")
            proposal = parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
            value = proposal.model_dump(mode="json")
            set_span_attributes(
                span,
                **{
                    "review_recovery.success": True,
                    "review_recovery.worker_kind": kind,
                    "review_recovery.downgrade_recommended": value.get(
                        "downgrade_recommended", False
                    ),
                },
            )
            set_span_output(span, value)
            return value, 1, None
        except Exception as error:  # one bounded advisory call; never loop here
            detail = f"{type(error).__name__}: {error}"
            set_span_attributes(
                span,
                **{
                    "review_recovery.success": False,
                    "review_recovery.worker_kind": kind,
                },
            )
            set_span_output(span, {"status": "FAILED", "error": detail})
            return None, 1, detail


async def resolve_recovery_human_decision(
    proposal: dict[str, Any] | None,
    *,
    event_id: str,
    user_request: str,
    current_step: Any,
) -> dict[str, Any] | None:
    """Pause for explicit user approval when a Reviewer proposes a downgrade."""

    if proposal is None or not proposal.get("downgrade_recommended", False):
        return proposal
    if proposal.get("human_decision") is not None:
        return proposal

    from feishu_exports import current_export_identity
    from recovery_approval import RECOVERY_APPROVALS, RecoveryApprovalRequest

    identity = current_export_identity()
    step_value = (
        current_step.model_dump(mode="json")
        if hasattr(current_step, "model_dump")
        else current_step
    )
    request = RecoveryApprovalRequest(
        event_id=str(event_id or "").strip() or "unknown-event",
        chat_id=(identity.chat_id if identity is not None else ""),
        step_id=int(
            getattr(current_step, "step_id", None)
            or (current_step.get("step_id") if isinstance(current_step, dict) else 0)
        ),
        worker_kind=str(
            getattr(current_step, "worker_kind", None)
            or (current_step.get("worker_kind") if isinstance(current_step, dict) else "")
        ).upper(),
        original_user_request=user_request,
        current_step=json.dumps(
            step_value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ),
        downgrade_reason=str(proposal.get("downgrade_reason") or ""),
        proposed_degraded_outcome=str(
            proposal.get("proposed_degraded_outcome") or ""
        ),
        user_confirmation_request=str(
            proposal.get("user_confirmation_request") or ""
        ),
    )
    with trace_span(
        "review_recovery.human_confirmation",
        kind="chain",
        input_value={
            "event_id": request.event_id,
            "step_id": request.step_id,
            "worker_kind": request.worker_kind,
        },
    ) as span:
        decision = await RECOVERY_APPROVALS.request_decision(request)
        result = {
            **proposal,
            "human_decision": decision.model_dump(mode="json"),
        }
        set_span_attributes(
            span,
            **{
                "review_recovery.human_decision": decision.status,
                "review_recovery.worker_kind": request.worker_kind,
            },
        )
        set_span_output(span, decision.model_dump(mode="json"))
        return result


__all__ = [
    "CodeRecoveryProposal",
    "GeneralRecoveryProposal",
    "WebRecoveryProposal",
    "build_recovery_proposal",
    "resolve_recovery_human_decision",
]
