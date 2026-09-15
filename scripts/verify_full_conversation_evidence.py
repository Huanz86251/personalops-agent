"""Read-only verification of a completed isolated full-conversation rehearsal."""
import json
from pathlib import Path
import sqlite3
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
folder = Path(sys.argv[1]).resolve()
result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
assert result["status"] == "returned"
with sqlite3.connect((ROOT/".agent/phoenix/phoenix.db").as_uri()+"?mode=ro", uri=True) as db:
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT s.*,t.trace_id FROM spans s JOIN traces t ON t.id=s.trace_rowid JOIN projects p ON p.id=t.project_rowid WHERE p.name=?",(result["project"],))]
assert len(rows) == result["span_count"]
for row in rows: row["attributes"] = json.loads(row["attributes"])
ids = {r["span_id"] for r in rows}
assert all(not r["parent_id"] or r["parent_id"] in ids for r in rows)
requests = [(r["attributes"].get("runtime") or {}).get("request_id") for r in rows]
assert not [k for k,v in Counter(requests).items() if k and v>1]
models = [r for r in rows if r["name"].startswith("LLM /")]
assert len(models) == result["model_calls"] == 8
assert not [r for r in rows if r["status_code"] == "ERROR"]
assert any(r["name"] == "Scheduler / Accept Code Review" for r in rows)
assert not any(r["name"] == "Scheduler / Review Web Step" for r in rows)
apps = list((folder/"workspace/conversations").glob("*/app.py"))
assert len(apps) == 1 and apps[0].read_text().strip() == "def add(a, b):\n    return a + b"
from workers.docker_sandbox import DockerCLI
cli = DockerCLI()
checkpoints = list((folder/"state/code_attempts").glob("*/runtime_checkpoint.json"))
assert len(checkpoints) == 1
checkpoint = json.loads(checkpoints[0].read_text(encoding="utf-8"))
assert checkpoint["phase"] == "TERMINAL"
pair = checkpoint["sandbox"]
containers = cli.run(["ps","-a","--format","{{.Names}}"],timeout=30)
volumes = cli.run(["volume","ls","--format","{{.Name}}"],timeout=30)
assert containers.returncode == volumes.returncode == 0
def names(output):
    return set((output.decode() if isinstance(output,bytes) else output).splitlines())
assert not {pair["worker_container"],pair["reviewer_container"]} & names(containers.stdout)
assert not {pair["candidate_volume"],pair["review_volume"]} & names(volumes.stdout)
report = {"project":result["project"],"spans":len(rows),"traces":len({r["trace_id"] for r in rows}),
    "model_calls":len(models),"missing_usage_requests":sum((r["attributes"].get("usage") or {}).get("status")=="missing" for r in models),
    "missing_parents":0,"duplicate_requests":0,"error_spans":0,"cleanup_verified":True,"delivered_file":str(apps[0]),
    "timings_ms":{r["name"]:(r["attributes"].get("timing") or {}).get("duration_ms") for r in rows if r["name"] in {"🎛️ Scheduler","🧩 Code Agent / Step 1","🛠️ Code Worker / Step 1 / Revision 1","🔍 Code Reviewer / Step 1 / Revision 1"}},
    "roots":[{"name":r["name"],"trace_id":r["trace_id"]} for r in rows if not r["parent_id"]]}
with (folder/"verified.json").open("x",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False))
