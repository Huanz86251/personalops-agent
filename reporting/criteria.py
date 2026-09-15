"""Harness-owned criterion identities; text remains in canonical reports."""
import json
from pydantic import BaseModel, Field, create_model
from planning_models import StepReport, CriterionStatus


def criterion_registry(criteria):
    return {f"C{i}": text for i, text in enumerate(criteria, 1)}


class CriterionReferenceResult(BaseModel):
    criterion_id: str = Field(description="Copy C1/C2 from the current Step contract. Never invent IDs.")
    evidence: list[str] = Field(default_factory=list, description="先引用已登记的E工具结果或A产物，再填写status；失败调用只能支持失败，不能支持成功。")
    status: CriterionStatus = Field(description="完成evidence后再判断MET/PARTIAL/NOT_MET/UNKNOWN。")


# Keep the established tool name, but do not ask the model to copy criterion prose.
ReferencedStepReport = create_model(
    "StepReport", __base__=StepReport,
    criterion_results=(list[CriterionReferenceResult], Field(default_factory=list)),
    evidence=(list[str], Field(default_factory=list, description="Registered E/A references only, no URLs or invented locators.")),
    errors=(list[str], Field(default_factory=list, description="Observed call/result error and its impact, with evidence reference. Not attempted is not an error; never infer API unavailability from missing evidence.")),
    unresolved_items=(list[str], Field(default_factory=list, description="For each remaining criterion: what is missing, whether attempted, and the prerequisite needed. Do not erase work already verified.")),
    next_action=(str | None, Field(default=None, description="Smallest next decision for Scheduler, based on actual remaining work. Do not privately restart General/Web or repeat successful writes.")),
    approved_artifact_refs=(list[str], Field(default_factory=list, max_length=12, description="Only registered A references whose content/use was verified. Harness publishes; reviewer does not invent shared paths.")),
)


def normalize_report(value, criteria, refs):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    value = dict(value)
    contract = criterion_registry(criteria)
    indexed = {}
    for raw in value.get("criterion_results", []):
        item = dict(raw)
        key = item.pop("criterion_id", None)
        # Read old checkpoints with exact original prose; never accept a paraphrase.
        if key is None:
            matches = [k for k, text in contract.items() if text == item.get("criterion")]
            if len(matches) != 1:
                raise ValueError("Use an unambiguous criterion_id from the current contract")
            key = matches[0]
        if key not in contract or key in indexed:
            raise ValueError(f"Unknown or duplicate criterion_id: {key}")
        item["criterion"] = contract[key]
        item["criterion_id"] = key
        indexed[key] = item
    if set(indexed) != set(contract):
        raise ValueError("Report must cover every current criterion exactly once")
    inverse = {short: canonical for canonical, short in refs.items()}
    def resolve(values):
        result = []
        for ref in values:
            if ref in inverse:
                ref = inverse[ref]
            elif ref not in refs:
                raise ValueError(f"Unknown evidence reference: {ref}")
            if ref not in result:
                result.append(ref)
        return result
    value["evidence"] = resolve(value.get("evidence", []))
    value["approved_artifact_refs"] = resolve(value.get("approved_artifact_refs", []))
    if any(not refs[ref].startswith("A") for ref in value["approved_artifact_refs"]):
        raise ValueError("Only registered artifact references can be approved")
    for item in indexed.values():
        item["evidence"] = resolve(item.get("evidence", []))
    value["criterion_results"] = [indexed[key] for key in contract]
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


def criterion_claim_repair_feedback(state, error):
    """Give the model the exact frozen IDs and a short model-facing repair shape."""
    contract = worker_registry(state)
    rows = [
        {"criterion_id": key, "evidence_tool_call_ids": [], "conclusion": "填写该项实际结果"}
        for key in contract
    ]
    return (
        "REPORT_SCHEMA_REJECTED\n"
        f"错误：{error}\n"
        "必须逐项覆盖冻结验收编号；先填 evidence_tool_call_ids，再填 conclusion。没有证据时保留空列表并如实写未完成，不能省略整张验收表。\n"
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
