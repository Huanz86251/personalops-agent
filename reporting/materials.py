"""Bounded read-only access to harness-registered review materials."""
import hashlib
import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


def build_reader(packet):
    materials = {}
    for attempt in packet.attempts:
        for item in attempt.resolved_artifacts:
            if item.review_ref:
                materials[item.review_ref] = ("file", item)
        for item in attempt.resolved_evidence:
            materials["evidence:" + item.tool_call_id] = ("evidence", item)
    catalog = [{"reference": "M" + str(i), "source": key, "kind": value[0]} for i, (key, value) in enumerate(materials.items(), 1)]
    materials = {"M" + str(i): value for i, value in enumerate(materials.values(), 1)}
    @tool
    def read_review_material(reference: str, offset: int = 0) -> str:
        """Read a registered candidate or evidence reference, 4000 characters per page. No arbitrary paths or writes. Use next_offset to continue; unavailable content is not verified."""
        if reference not in materials or offset < 0:
            raise ValueError("Unknown reference or negative offset")
        kind, item = materials[reference]
        if kind == "evidence":
            text = item.result
        else:
            if not item.storage_path:
                return json.dumps({"status":"UNAVAILABLE", "reason":"No registered local content; do not approve based on existence alone."})
            path = Path(item.storage_path)
            if any(x.is_symlink() for x in [path, *path.parents]) or not path.is_file():
                raise ValueError("Candidate is not a regular registered file")
            if path.stat().st_size > 10_000_000:
                return json.dumps({"status":"UNAVAILABLE", "reason":"File exceeds read limit"})
            data = path.read_bytes()
            if item.sha256 and hashlib.sha256(data).hexdigest() != item.sha256:
                raise ValueError("Candidate changed since registration")
            try: text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                return json.dumps({"status":"UNAVAILABLE", "reason":"Not UTF-8 text; obtain a verified text representation"})
        end = offset + 4000
        return json.dumps({"reference":reference, "content":text[offset:end], "next_offset":end if end < len(text) else None}, ensure_ascii=False)
    return read_review_material, catalog


class ReviewWithReads:
    def __init__(self, model, final_model, packet, rounds):
        self.reader, self.references = build_reader(packet)
        self.model = model
        self.final_model = final_model
        self.rounds = rounds
        self.used = 0

    async def ainvoke(self, messages):
        from reporting.criteria import ReferencedStepReport as StepReport
        self.used += 1
        # Reserve the final request for an ordinary structured report.
        if self.used >= max(1, self.rounds - 1):
            from prompt_loader import load_prompt
            messages.append({"role": "user", "content": load_prompt("reporters/finalize_step_report")})
            return await self.final_model.ainvoke(messages)
        response = await self.model.bind_tools([self.reader, StepReport]).ainvoke(messages)
        calls = getattr(response, "tool_calls", [])
        final = [c for c in calls if c["name"] == "StepReport"]
        if final and len(calls) == 1:
            return {"parsed":final[0]["args"], "raw":response}
        if calls and all(c["name"] == self.reader.name for c in calls):
            messages.append(response)
            for call in calls:
                try:
                    from observability import trace_span, set_span_output
                    with trace_span("Reviewer / Read Material", kind="tool", input_value=call["args"]) as span:
                        result = await self.reader.ainvoke(call["args"])
                        set_span_output(span, result)
                    status = "success"
                except Exception as error:
                    result = str(error); status = "error"
                messages.append(ToolMessage(content=result, tool_call_id=call["id"], status=status))
            return {"material_read": True}
        return {"parsing_error":"Submit one StepReport or read registered materials; do not mix the two."}
