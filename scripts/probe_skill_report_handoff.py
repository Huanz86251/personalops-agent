"""Replay saved real Web observations through bounded review and Scheduler facts.

No network/model calls. Reports are manually authored fixtures, not model output.
Evidence is input explicitly; outputs always go to a new Git-ignored directory.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning_models import PlanStep, StepReport
from reporting import build_step_review_packet
from scheduler_runtime import SchedulerConversation, record_graph_facts
from workers.submission import WorkerSubmissionRecord


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("whole_page", type=Path)
    parser.add_argument("targeted_browser_result", type=Path)
    args = parser.parse_args()
    whole = json.loads(args.whole_page.read_text(encoding="utf-8"))
    targeted = json.loads(args.targeted_browser_result.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1] / ".agent" / "skill-policy-handoffs"
    folder = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc)
    observations = {}
    for name, evidence in (("whole", whole), ("targeted", targeted)):
        observations[name] = json.dumps(evidence["result"], ensure_ascii=False)
    assert "参考报价" in observations["whole"] and "参考报价" in observations["targeted"]
    for label, criterion in (("reference", "给出M650-M公开参考报价及来源，不要求店铺现价"),
                             ("seller", "核实淘宝指定店铺M650-M选定SKU当前实际价格")):
        step = PlanStep(step_id=1, objective=criterion, success_criteria=[criterion], worker_kind="WEB")
        packets = {}
        for name, text in observations.items():
            original = whole if name == "whole" else targeted
            record = WorkerSubmissionRecord.model_validate(dict(submitted_at=now,
                worker_id="probe-web", event_id="policy-probe", step_id="1", total_tool_calls=1,
                submission=dict(summary="找到M650-M参考报价", final_conclusion="参考价不是指定店现价",
                    criterion_claims=[dict(criterion=criterion, conclusion="公开参考价269元；指定店现价未核实",
                        evidence_tool_call_ids=[name])]),
                resolved_evidence=[dict(tool_call_id=name, tool_name="fetch_webpage" if name == "whole" else "browser_find",
                    arguments=original.get("input", original.get("call", {}).get("arguments")),
                    result=text, result_chars=len(text))]))
            packet = build_step_review_packet(user_request=criterion, plan_objective=criterion, current_step=step,
                current_attempt={"attempt": 1, "worker_submission": record.model_dump(mode="json"),
                                 "messages": ["PRIVATE_WORKER_CONTEXT"]}, stop_reason="提交已有证据")
            packets[name] = packet
            (folder / f"{label}-{name}-packet.json").write_text(packet.model_dump_json(indent=2), encoding="utf-8")
        assert "参考报价" not in packets["whole"].attempts[0].resolved_evidence[0].result
        assert "参考报价" in packets["targeted"].attempts[0].resolved_evidence[0].result
        fulfilled = label == "reference"
        report = StepReport(step_id=1, status="COMPLETED" if fulfilled else "PARTIAL",
            summary="已核实公开参考报价；它不代表淘宝指定店现价",
            stop_reason="已审核本次提交的来源证据", assessment_source="INDEPENDENT_REVIEW",
            criterion_results=[dict(criterion=criterion, status="MET" if fulfilled else "UNKNOWN", evidence=["targeted: browser_find 参考报价"] )],
            confirmed_results=["ZOL M650-M页面展示参考报价269元"], completed_work=["定位参数页并读取参考价片段"],
            artifacts=[], approved_artifact_refs=[], worker_contributions=[dict(worker_id="probe-web", contribution="取得公开参考价")],
            evidence=["targeted", "https://detail.zol.com.cn/1970/1969107/param.shtml"], errors=[],
            unresolved_items=[] if fulfilled else ["指定淘宝店卖家、SKU、当前报价与优惠条件"],
            next_action=None if fulfilled else "保留已核实参考价；请提供目标店商品链接/规格及价格截图后核对卖家、SKU与优惠，不重查参考价。",
            request_replan=False, replan_reason=None)
        # This report is an explicit manual assessment fixture. No inference tool ran.
        (folder / f"{label}-manual-report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        session = SchedulerConversation({}, None, threshold=1)
        record_graph_facts(session, {"completed_step_reports": [report.model_dump(mode="json")]})
        session.compact()
        delivered = json.loads(session.wire()[0]["content"])["StepReport"]
        assert delivered == report.model_dump(mode="json")
        assert "PRIVATE_WORKER_CONTEXT" not in json.dumps(session.wire())
        (folder / f"{label}-scheduler-facts.json").write_text(json.dumps(session.wire(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(folder), "result": "PASS", "checks": [
        "Real whole-page reference label omitted by bounded packet", "Real browser_find quote remains visible",
        "Manual reference/current-seller assessments differ", "Every StepReport field survives Scheduler delivery",
        "No Worker conversation included"], "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
