"""Read-only persisted trace and prompt-size analysis for the offline rehearsal."""
import collections
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skill_runtime import load_catalog
from prompt_loader import load_prompt

base = ROOT / ".agent/chain-audit/20260906"
summaries = [json.loads((base / name / "summary.json").read_text()) for name in ["scripted-routes-final"]]
families = {r["family"]: r for s in summaries for r in s["families"]}
requests = []
for name in ["scripted-routes-final"]:
    requests += [json.loads(line) for line in (base / name / "requests.jsonl").read_text(encoding="utf-8").splitlines()
                ]
catalog = load_catalog()
report = {"families": list(families.values()), "skills": [{"name": a["name"], "chars": len(a["content"]), "roles": a["roles"]} for a in catalog], "traces": [], "requests": []}
with sqlite3.connect((ROOT / ".agent/phoenix/phoenix.db").as_uri() + "?mode=ro", uri=True) as db:
    db.row_factory = sqlite3.Row
    for family, run in families.items():
        spans = [dict(s) for s in db.execute("SELECT s.* FROM spans s JOIN traces t ON t.id=s.trace_rowid WHERE t.trace_id=? ORDER BY s.start_time", (run["trace_id"],))]
        for s in spans:
            if isinstance(s.get("attributes"), str):
                s["attributes"] = json.loads(s["attributes"])
        (base / "scripted-routes-boundaries" / (family + "-spans.private.json")).write_text(json.dumps(spans, ensure_ascii=False, indent=2), encoding="utf-8")
        ids = {s["span_id"] for s in spans}
        report["traces"].append({"family": family, "count": len(spans),
            "roots": sum(not s.get("parent_id") for s in spans),
            "missing_parents": sum(bool(s.get("parent_id")) and s["parent_id"] not in ids for s in spans),
            "names": dict(collections.Counter(s["name"] for s in spans)),
            "max_attribute_chars": max((len(json.dumps(s["attributes"], ensure_ascii=False)) for s in spans), default=0)})
for r in requests:
    report["requests"].append({k: r[k] for k in ["case", "role", "input_chars", "approx_message_tokens", "schema_chars"]})
report["prompts"] = {name: len(load_prompt(name)) for name in ["workers/code_worker", "workers/web_worker", "reviewers/code", "planning/scheduler", "reporters/step_report", "routing/skill_selector"]}
(base / "scripted-analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
