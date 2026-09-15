"""Read an already downloaded private attachment in isolated diagnostic state.

No mailbox/network/model calls. MANUAL import is not a production bridge. Verify
the real Scheduler fact path with an explicitly manual PARTIAL StepReport.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_email_attachment_read import payload
from mcp_runtime import EMAIL_ATTACHMENT_PATH
from deepagents.backends.utils import create_file_data, file_data_to_string
from tools.local_native import attachment_to_text
from planning_models import StepReport
from scheduler_runtime import SchedulerConversation, record_graph_facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download_record", type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.download_record.read_text(encoding="utf-8"))
    assert evidence["tool"] == "email_download_attachment"
    attachment = payload(evidence["result"])["attachment"]
    path = Path(attachment["path"]).resolve()
    path.relative_to(EMAIL_ATTACHMENT_PATH.resolve())
    data = path.read_bytes()
    assert len(data) == attachment["size"] <= 2 * 1024 * 1024
    assert hashlib.sha256(data).hexdigest() == attachment["sha256"]
    folder = args.download_record.parent / ("handoff-" + datetime.now(timezone.utc).strftime("%H%M%S") + "-" + uuid4().hex[:6])
    folder.mkdir(exist_ok=False)
    key = "/inputs/attachment" + path.suffix.lower()
    file_record = create_file_data(base64.b64encode(data).decode("ascii"))
    file_record["encoding"] = "base64"
    state = {"worker_id": "manual-email-diagnostic", "files": {key: file_record}}
    result = attachment_to_text.func(path=key, output_path="/artifacts/read.md", runtime=SimpleNamespace(state=state, tool_call_id="diagnostic-ocr"), ocr="auto")
    (folder / "conversion.json").write_text(json.dumps(result.update, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    state["files"].update(result.update["files"])
    text = file_data_to_string(state["files"]["/artifacts/read.md"])
    metadata = json.loads(file_data_to_string(state["files"]["/artifacts/read.json"]))
    summary = metadata["summary"]
    report = StepReport(step_id=1, status="PARTIAL",
        summary="真实附件已下载；现有邮箱路径未接入Worker任务文件空间",
        stop_reason="宿主下载路径被任务文件阅读器拒绝", assessment_source="GENERAL_SELF_REPORT",
        criterion_results=[
            dict(criterion="下载一份真实邮件附件并校验", status="MET", evidence=["email_download_attachment返回与磁盘SHA256一致"]),
            dict(criterion="当前生产链路读取附件并向Scheduler交接内容", status="NOT_MET", evidence=["Use a virtual file path, not a host path or URL."])],
        confirmed_results=[f"已下载PDF {len(data)}字节并校验SHA256", "下载回执不包含任务文件导入；未自动交接附件内容"],
        completed_work=["在有界近期邮件窗口定位一份附件并下载"], artifacts=[], approved_artifact_refs=[],
        worker_contributions=[dict(worker_id="manual-email-diagnostic", contribution="实测下载与读取接口边界；报告由测试人员手工编写")],
        evidence=["live download record", "host-path rejection"], errors=["附件宿主路径不能直接供任务文件阅读器读取"],
        unresolved_items=["运行时缺少邮箱附件到任务文件空间的受控导入"],
        next_action="先由宿主补受控附件导入及运行归属，再重读并提交带页码的内容证据；不重复下载、不发送。",
        request_replan=False, replan_reason=None)
    session = SchedulerConversation({}, None, threshold=1)
    record_graph_facts(session, {"completed_step_reports": [report.model_dump(mode="json")]})
    session.compact()
    received = json.loads(session.wire()[0]["content"])["StepReport"]
    assert received == report.model_dump(mode="json")
    assert text not in json.dumps(session.wire(), ensure_ascii=False)
    (folder / "manual-report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (folder / "scheduler-facts.json").write_text(json.dumps(session.wire(), ensure_ascii=False, indent=2), encoding="utf-8")
    outcome = {"folder": str(folder), "manual_import_only": True, "reader_summary": summary,
               "markdown_characters": len(text), "scheduler_received_report": True,
               "scheduler_auto_received_body": False, "production_bridge_missing": True, "model_calls": 0}
    (folder / "summary.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outcome, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
