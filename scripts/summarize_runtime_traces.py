"""Read persisted private rehearsal spans and check tree/usage ownership."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import argparse
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, help="Exact Phoenix project name to inspect")
args = parser.parse_args()
out = root / ".agent/trace-hierarchy" / ("persisted-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
out.mkdir(parents=True, exist_ok=True)
with sqlite3.connect((root / ".agent/phoenix/phoenix.db").as_uri() + "?mode=ro", uri=True) as db:
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT s.*, t.trace_id FROM spans s JOIN traces t ON t.id=s.trace_rowid JOIN projects p ON p.id=t.project_rowid WHERE p.name=? ORDER BY s.start_time", (args.project,))]
assert rows, "No spans found for requested project"
for row in rows:
    row["attributes"] = json.loads(row["attributes"]) if isinstance(row["attributes"], str) else row["attributes"]
ids = {s["span_id"] for s in rows}
missing = [s["span_id"] for s in rows if s.get("parent_id") and s["parent_id"] not in ids]
request_ids = [(s["attributes"].get("runtime") or {}).get("request_id") for s in rows]
duplicates = [key for key, count in Counter(request_ids).items() if key and count > 1]
report = {"project": args.project, "spans": len(rows), "traces": len({s["trace_id"] for s in rows}),
          "missing_parents": missing, "duplicate_request_ids": duplicates,
          "names": dict(Counter(s["name"] for s in rows)),
          "roots": [{"name": s["name"], "trace_id": s["trace_id"], "status": s["status_code"]} for s in rows if not s.get("parent_id")]}
(out / "spans.private.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
(out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
assert not missing and not duplicates
print(json.dumps(report, ensure_ascii=False, indent=2))
