"""Replay saved invalid plans locally; never constructs a provider client."""
import asyncio
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_planning_repair_prefix import Scripted
from hard_planning import _invoke_structured
from planning_models import SupervisorDecision
from langchain_core.messages import AIMessage


async def main():
    source = ROOT / ".agent/chain-audit/20260906/paid-thinking-off/events.private.jsonl"
    results = []
    for event in map(json.loads, source.read_text(encoding="utf-8").splitlines()):
        if event["event"] != "model_end" or (event.get("record") or {}).get("call_index") not in {2, 3}:
            continue
        raw = event["response"]["generations"][0][0]["text"]
        plan = json.loads(raw)
        fixed = copy.deepcopy(plan)
        for step in fixed["steps"]:
            if step["worker_kind"] == "CODE":
                step.pop("artifact_outputs", None)
            else:
                for artifact in step.get("artifact_outputs", []):
                    if artifact.get("disposition") == "INTERNAL_HANDOFF":
                        artifact.pop("target_path", None)
        valid = SupervisorDecision.model_validate(fixed)
        model = Scripted([{"raw": AIMessage(content=raw), "parsing_error": ValueError("Failed to parse " + raw + " errors at end")}, {"parsed": valid}])
        result = await _invoke_structured(model, prompt="Stable offline replay prefix", output_schema=SupervisorDecision,
            trace_name="offline replay", fallback_factory=lambda error: (_ for _ in ()).throw(AssertionError(error)))
        feedback = model.requests[1][-1]["content"]
        assert "INTERNAL_HANDOFF artifact不得提供target_path" in feedback
        assert "CODE Step使用code_task" in feedback
        assert model.requests[1][:-2] == model.requests[0]
        assert model.requests[1][-2]["content"] == raw
        assert not result.used_fallback
        results.append({"original_call": event["record"]["call_index"], "prefix_equal": True,
                        "raw_plan_equal": True, "repair_feedback": feedback, "model_rounds": result.model_rounds_used,
                        "workers": [s.worker_kind for s in result.output.steps], "paid_calls": 0})
    assert len(results) == 2
    out = ROOT / ".agent/chain-audit/20260906/repair-prefix-replay.json"
    with out.open("x", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print("Both saved plans repaired by scripted responses; exact prefix retained; 0 paid calls.")


if __name__ == "__main__":
    asyncio.run(main())
