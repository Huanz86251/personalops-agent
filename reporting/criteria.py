"""Harness-owned criterion identities; text remains in canonical reports."""
import json
from pydantic import BaseModel, Field, create_model, field_validator
from pydantic.json_schema import SkipJsonSchema
from planning_models import StepReport, CriterionStatus
from schema_normalization import EvidenceRefText, ReportItem, coerce_text_sequence, derive_step_status


def criterion_registry(criteria):
    return {f"C{i}": text for i, text in enumerate(criteria, 1)}


class CriterionReferenceResult(BaseModel):
    criterion_id: str = Field(description="Copy C1/C2 from the current Step contract. Never invent IDs.")
    evidence: list[EvidenceRefText] = Field(default_factory=list, max_length=24, description="先引用已登记的E普通工具结果、W开头的具体网页结果或A产物，最多24个，再填写status；失败调用只能支持失败，不能支持成功。")
    status: CriterionStatus = Field(description="完成evidence后再判断MET/PARTIAL/NOT_MET/UNKNOWN。")

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value):
        return coerce_text_sequence(value)


# Keep the established tool name, but do not ask the model to copy criterion prose.
ReferencedStepReport = create_model(
    "StepReport", __base__=StepReport,
    status=(SkipJsonSchema[str], Field(
        default="BLOCKED",
        description="Harness根据criterion_results确定；模型不要填写。",
    )),
    criterion_results=(list[CriterionReferenceResult], Field(default_factory=list, max_length=24)),
    evidence=(list[EvidenceRefText], Field(default_factory=list, max_length=48, description="最多48个已登记E/W/A编号；不要填写URL或自造定位符。")),
    errors=(list[ReportItem], Field(default_factory=list, max_length=24, description="最多24项、每项最多2000字符；记录真实调用错误和影响。")),
    unresolved_items=(list[ReportItem], Field(default_factory=list, max_length=24, description="最多24项、每项最多2000字符；写缺项、是否尝试及必要前提。")),
    next_action=(str | None, Field(default=None, max_length=4000, description="Scheduler的最小下一步，最多4000字符；不要私自重启Worker。")),
    approved_artifact_refs=(list[str], Field(default_factory=list, max_length=12, description="Only registered A references whose content/use was verified. Harness publishes; reviewer does not invent shared paths.")),
)


def normalize_report(value, criteria, refs, aliases=None, *, mode="strict"):
    if mode not in {"strict", "balanced"}:
        raise ValueError("review evidence mode must be strict or balanced")
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    value = dict(value)
    contract = criterion_registry(criteria)
    indexed = {}
    normalization_notes = []
    invalid_criterion_ids = set()
    for raw in value.get("criterion_results", []):
        item = dict(raw)
        key = item.pop("criterion_id", None)
        # Read old checkpoints with exact original prose; never accept a paraphrase.
        if key is None:
            matches = [k for k, text in contract.items() if text == item.get("criterion")]
            if len(matches) != 1:
                if mode == "strict":
                    raise ValueError("Use an unambiguous criterion_id from the current contract")
                normalization_notes.append("ignored criterion row without an unambiguous current criterion_id")
                continue
            key = matches[0]
        if key not in contract:
            if mode == "strict":
                raise ValueError(f"Unknown or duplicate criterion_id: {key}")
            normalization_notes.append(f"ignored unknown criterion_id {key}")
            continue
        if key in indexed or key in invalid_criterion_ids:
            if mode == "strict":
                raise ValueError(f"Unknown or duplicate criterion_id: {key}")
            indexed.pop(key, None)
            invalid_criterion_ids.add(key)
            normalization_notes.append(f"replaced duplicate criterion_id {key} with UNKNOWN")
            continue
        item["criterion"] = contract[key]
        item["criterion_id"] = key
        indexed[key] = item
    if set(indexed) != set(contract):
        if mode == "strict":
            raise ValueError("Report must cover every current criterion exactly once")
        for key, text in contract.items():
            if key not in indexed:
                indexed[key] = {
                    "criterion": text,
                    "criterion_id": key,
                    "status": "UNKNOWN",
                    "evidence": [],
                }
                normalization_notes.append(f"synthesized missing criterion_id {key} as UNKNOWN")
    inverse = {short: canonical for canonical, short in refs.items()}
    def resolve(values, *, field_name):
        result = []
        for ref in values:
            ref = (aliases or {}).get(ref, ref)
            if ref in inverse:
                ref = inverse[ref]
            elif ref not in refs:
                if mode == "strict":
                    raise ValueError(f"Unknown evidence reference: {ref}")
                normalization_notes.append(f"dropped unknown {field_name} reference {ref}")
                continue
            if ref not in result:
                result.append(ref)
        return result
    value["evidence"] = resolve(value.get("evidence", []), field_name="evidence")
    value["approved_artifact_refs"] = resolve(
        value.get("approved_artifact_refs", []), field_name="artifact"
    )
    invalid_approvals = [
        ref for ref in value["approved_artifact_refs"]
        if not refs[ref].startswith("A")
    ]
    if invalid_approvals:
        if mode == "strict":
            raise ValueError("Only registered artifact references can be approved")
        value["approved_artifact_refs"] = [
            ref for ref in value["approved_artifact_refs"]
            if ref not in invalid_approvals
        ]
        normalization_notes.append("dropped non-artifact approval references")
    unresolved = list(value.get("unresolved_items", []) or [])
    for key, item in indexed.items():
        item["evidence"] = resolve(
            item.get("evidence", []), field_name=f"criterion {key} evidence"
        )
        if mode == "balanced" and item.get("status") == "MET" and not item["evidence"]:
            item["status"] = "UNKNOWN"
            if contract[key] not in unresolved:
                unresolved.append(contract[key])
            normalization_notes.append(f"downgraded unsupported MET criterion {key} to UNKNOWN")
    value["criterion_results"] = [indexed[key] for key in contract]
    if value["criterion_results"]:
        value["status"] = derive_step_status(
            item.get("status", "UNKNOWN")
            for item in value["criterion_results"]
        )
        if (
            value["status"] == "BLOCKED"
            and any(note.startswith("downgraded unsupported MET") for note in normalization_notes)
        ):
            # The attempt produced a claim but its citation was unusable. This
            # is partial progress, unlike a wholly absent/unattempted report.
            value["status"] = "PARTIAL"
    else:
        value["status"] = value.get("status") or "BLOCKED"
    if mode == "balanced":
        value["unresolved_items"] = unresolved
        if normalization_notes:
            errors = list(value.get("errors", []) or [])
            errors.append(
                "Harness balanced normalization: " + "; ".join(normalization_notes)
            )
            value["errors"] = errors
    return StepReport.model_validate(value)


def worker_registry(state):
    registry = state.get("worker_criterion_refs", {})
    if registry:
        return dict(registry)
    for message in [*state.get("worker_archived_messages", []), *state.get("messages", [])]:
        role = message.get("role", message.get("type")) if isinstance(message, dict) else getattr(message, "type", None)
        if role not in {"human", "user"}:
            continue
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, str):
            for line in content.splitlines():
                if line.startswith("HARNESS_CRITERIA: "):
                    registry = json.loads(line.removeprefix("HARNESS_CRITERIA: "))
    return registry



def validate_worker_claim_coverage(state, claims):
    """Require one claim for every frozen criterion before a Worker can close."""
    contract = worker_registry(state)
    if not contract:
        return
    identities = [getattr(claim, "criterion_id", None) for claim in claims]
    missing = [key for key in contract if key not in identities]
    unknown = [key for key in identities if key not in contract]
    duplicates = sorted({key for key in identities if key and identities.count(key) > 1})
    if missing or unknown or duplicates:
        raise ValueError(
            "criterion_claims must cover every HARNESS_CRITERIA ID exactly once; "
            f"missing={missing}, unknown={unknown}, duplicates={duplicates}"
        )


def complete_missing_worker_claims(state, claims):
    """Fill omitted frozen criteria without inventing success or evidence.

    Missing rows are a form-shape defect, not a reason to rerun business tools
    or repeatedly ask the model to repair the same table.  Unknown and
    duplicate identities remain hard errors because they cannot be mapped
    deterministically to the frozen contract.
    """
    from workers.submission import WorkerCriterionClaim

    contract = worker_registry(state)
    if not contract:
        return list(claims)
    identities = [getattr(claim, "criterion_id", None) for claim in claims]
    unknown = [key for key in identities if key not in contract]
    duplicates = sorted({key for key in identities if key and identities.count(key) > 1})
    if unknown or duplicates:
        raise ValueError(
            "criterion_claims contains unknown or duplicate HARNESS_CRITERIA IDs; "
            f"unknown={unknown}, duplicates={duplicates}"
        )
    indexed = {
        claim.criterion_id: claim
        for claim in claims
        if claim.criterion_id in contract
    }
    return [
        indexed.get(key)
        or WorkerCriterionClaim(
            criterion_id=key,
            criterion=text,
            evidence_tool_call_ids=[],
            conclusion=(
                "Worker未提交本项判断或证据；Harness按UNKNOWN保守处理。"
            ),
            status="UNKNOWN",
        )
        for key, text in contract.items()
    ]


def criterion_claim_repair_feedback(state, error):
    """Give the model the exact frozen IDs and a short model-facing repair shape."""
    contract = worker_registry(state)
    rows = [
        {"criterion_id": key, "evidence_tool_call_ids": [], "conclusion": "填写该项实际结果", "status": "UNKNOWN"}
        for key in contract
    ]
    return (
        "REPORT_SCHEMA_REJECTED\n"
        f"错误：{error}\n"
        "必须逐项覆盖冻结验收编号；先填 evidence_tool_call_ids，再填 conclusion 和 status。没有证据时保留空列表并如实写未完成，不能省略整张验收表。\n"
        f"criterion_claims 最小示例：{json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}\n"
        "保留已完成的业务结果，只重新提交报告；不要重新调用业务API。"
    )


def restore_claims(args, contract):
    if isinstance(args, list):
        return [restore_claims(x, contract) for x in args]
    if not isinstance(args, dict):
        return args
    result = {k: restore_claims(v, contract) for k, v in args.items()}
    if result.get("criterion_id") is not None:
        key = result.pop("criterion_id")
        if key not in contract:
            raise ValueError(f"Unknown criterion_id: {key}")
        result["criterion"] = contract[key]
        result["criterion_id"] = key
    claims = result.get("criterion_claims")
    if claims is not None and contract:
        texts = [claim.get("criterion") for claim in claims]
        identities = [claim.get("criterion_id") or claim.get("criterion") for claim in claims]
        if len(identities) != len(set(identities)) or any(text not in contract.values() for text in texts):
            raise ValueError("Duplicate or unknown criterion claim")
    return result
