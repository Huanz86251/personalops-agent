"""Copy saved trace evidence into a clearly labelled Phoenix preview. No model/world calls."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_overview import build_overview, tool_purpose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = ROOT / ".agent/trace-previews" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    saved = json.loads((source / "phoenix.private.json").read_text(encoding="utf-8"))["data"]
    task = json.loads((source / "task.private.json").read_text(encoding="utf-8"))
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    raw = json.loads((source / "spans.private.json").read_text(encoding="utf-8"))
    overview, context = build_overview(task, result, raw, preview=True)
    (out / "overview.private.md").write_text(overview, encoding="utf-8")
    (out / "context.private.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    # Rebuild a new trace preserving the original tree. Original traces are immutable.
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import KeyValue, AnyValue
    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add()
    project = "Trace 阅读预览 · " + datetime.now().strftime("%m-%d %H:%M")
    resource.resource.attributes.append(KeyValue(key="openinference.project.name",value=AnyValue(string_value=project)))
    scope = resource.scope_spans.add()
    scope.scope.name = "personalops.trace-preview"
    ids = {r["context"]["span_id"]:uuid4().hex[:16] for r in saved}
    trace_id = uuid4().hex
    earliest = min(datetime.fromisoformat(r["start_time"]).timestamp() for r in saved)
    offset = time.time() - earliest - 120
    counts = Counter()
    root_id = None
    for row in sorted(saved, key=lambda r:r["start_time"]):
        span = scope.spans.add()
        span.trace_id = bytes.fromhex(trace_id)
        span.span_id = bytes.fromhex(ids[row["context"]["span_id"]])
        if row["parent_id"]:
            span.parent_span_id = bytes.fromhex(ids[row["parent_id"]])
        span.start_time_unix_nano = int((datetime.fromisoformat(row["start_time"]).timestamp()+offset)*1e9)
        span.end_time_unix_nano = int((datetime.fromisoformat(row["end_time"]).timestamp()+offset)*1e9)
        attrs = dict(row["attributes"])
        attrs.update({"audit.synthetic":True,"audit.preview":True,"audit.source_trace_id":row["context"]["trace_id"],
                      "audit.note":"历史证据展示回放；没有新任务执行，不计入准确率。耗时来自历史录制。",
                      "session.id":"preview-"+trace_id})
        name = row["name"]
        kind = row["span_kind"]
        if kind in {"LLM", "TOOL"}:
            if kind == "TOOL":
                name = tool_purpose(attrs.get("tool.name",name.removeprefix("Tool / ")),attrs.get("input.value"))
            key = (row["parent_id"],name)
            counts[key] += 1
            name += f" · {counts[key]:02d}"
        if not row["parent_id"]:
            name = "阅读预览 · 任务总览（非新评测）"
            root_id = ids[row["context"]["span_id"]]
            attrs.update({"input.value":json.dumps(context,ensure_ascii=False),"input.mime_type":"application/json",
                          "output.value":overview,"output.mime_type":"text/plain"})
        span.name = name
        span.status.code = 2 if row["status_code"]=="ERROR" else 1
        attrs["openinference.span.kind"] = kind
        for key, value in attrs.items():
            item = span.attributes.add();item.key=key
            if isinstance(value,bool):item.value.bool_value=value
            elif isinstance(value,int):item.value.int_value=value
            elif isinstance(value,float):item.value.double_value=value
            elif isinstance(value,(list,tuple)) and all(isinstance(v,str) for v in value):
                item.value.array_value.values.extend(AnyValue(string_value=v) for v in value)
            else:item.value.string_value=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
    from trace_chat import chat_attributes
    originals = list(scope.spans)
    for original in originals:
        attrs = {a.key: a.value.string_value for a in original.attributes}
        if attrs.get("openinference.span.kind") != "LLM":
            continue
        incoming = json.loads(attrs.get("input.value", "{}"))
        outgoing = json.loads(attrs.get("output.value", "{}"))
        for key, value in chat_attributes(incoming, outgoing).items():
            if key.startswith("llm."):
                item = next((a for a in original.attributes if a.key == key), None)
                if item is None: item = original.attributes.add(); item.key = key
                item.value.string_value = value
        view = scope.spans.add()
        view.trace_id = original.trace_id
        view.span_id = bytes.fromhex(uuid4().hex[:16])
        view.parent_span_id = original.span_id
        view.start_time_unix_nano = original.end_time_unix_nano
        view.end_time_unix_nano = original.end_time_unix_nano
        view.name = "Messages / 完整对话"
        view.status.code = 1
        values = {**chat_attributes(incoming, outgoing),
                  "openinference.span.kind": "CHAIN", "audit.preview": True,
                  "input.value": json.dumps(incoming, ensure_ascii=False), "input.mime_type": "application/json",
                  "output.value": json.dumps(outgoing, ensure_ascii=False), "output.mime_type": "application/json"}
        for key, value in values.items():
            item = view.attributes.add(); item.key = key
            if isinstance(value, bool): item.value.bool_value = value
            else: item.value.string_value = value
    packet = request.SerializeToString()
    req = urllib.request.Request("http://127.0.0.1:6007/v1/traces",data=packet,headers={"Content-Type":"application/x-protobuf"})
    with urllib.request.urlopen(req,timeout=20) as response:
        assert response.status==200
    manifest = {"project":project,"trace_id":trace_id,"root_span_id":root_id,"span_count":len(scope.spans),
                "source":str(source),"preview_only":True,"paid_calls":0,"new_task_runs":0}
    (out / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(out),**manifest},ensure_ascii=False))


if __name__ == "__main__":main()
