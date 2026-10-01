"""Bounded read-only access to harness-registered review materials."""
import hashlib
import json
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


MATERIAL_PAGE_CHARS = 4000
BATCH_MATERIAL_MAX_ITEMS = 12
BATCH_MATERIAL_MAX_CHARS = 32000


class MaterialReadRequest(BaseModel):
    reference: str = Field(description="Registered M/E/A material reference.")
    offset: int = Field(default=0, ge=0)


def build_material_reader(packet, reference_aliases=None):
    source_materials = {}
    for attempt in packet.attempts:
        for item in attempt.resolved_artifacts:
            if item.review_ref:
                source_materials[item.review_ref] = ("file", item)
        for item in attempt.resolved_evidence:
            source_materials["evidence:" + item.tool_call_id] = ("evidence", item)
    catalog = [{"reference": "M" + str(i), "source": key, "kind": value[0]}
               for i, (key, value) in enumerate(source_materials.items(), 1)]
    materials = {"M" + str(i): value
                 for i, value in enumerate(source_materials.values(), 1)}
    for alias, source in (reference_aliases or {}).items():
        if source in source_materials:
            materials[alias] = source_materials[source]
    def read_one(reference: str, offset: int = 0) -> dict:
        """Read one bounded page without exposing arbitrary filesystem paths."""
        if reference not in materials or offset < 0:
            raise ValueError("Unknown reference or negative offset")
        kind, item = materials[reference]
        if kind == "evidence":
            text = item.result
        else:
            if not item.storage_path:
                return {"reference": reference, "status":"UNAVAILABLE", "reason":"No registered local content; do not approve based on existence alone."}
            path = Path(item.storage_path)
            if any(x.is_symlink() for x in [path, *path.parents]) or not path.is_file():
                raise ValueError("Candidate is not a regular registered file")
            if path.stat().st_size > 10_000_000:
                return {"reference": reference, "status":"UNAVAILABLE", "reason":"File exceeds read limit"}
            data = path.read_bytes()
            if item.sha256 and hashlib.sha256(data).hexdigest() != item.sha256:
                raise ValueError("Candidate changed since registration")
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                return {"reference": reference, "status":"UNAVAILABLE", "reason":"Not UTF-8 text; obtain a verified text representation"}
        end = offset + MATERIAL_PAGE_CHARS
        return {"reference":reference, "content":text[offset:end], "next_offset":end if end < len(text) else None}

    @tool
    def read_review_materials(requests: list[MaterialReadRequest]) -> str:
        """Read 1-12 registered evidence/artifact pages in one batch. Use a one-item batch for pagination; each result keeps its reference, content and next_offset. Arbitrary paths and writes are forbidden."""
        if not requests:
            raise ValueError("At least one registered material is required")
        if len(requests) > BATCH_MATERIAL_MAX_ITEMS:
            raise ValueError(f"At most {BATCH_MATERIAL_MAX_ITEMS} materials may be read per call")
        pages = []
        total_chars = 0
        seen = set()
        for request in requests:
            key = (request.reference, request.offset)
            if key in seen:
                continue
            seen.add(key)
            page = read_one(request.reference, request.offset)
            rendered = json.dumps(page, ensure_ascii=False)
            if pages and total_chars + len(rendered) > BATCH_MATERIAL_MAX_CHARS:
                pages.append({
                    "reference": request.reference,
                    "offset": request.offset,
                    "status": "DEFERRED",
                    "reason": "Batch character limit reached; request this page in the next batch.",
                })
                continue
            pages.append(page)
            total_chars += len(rendered)
        return json.dumps({"materials": pages}, ensure_ascii=False)

    return read_review_materials, catalog


class ReviewWithReads:
    def __init__(self, model, final_model, packet, rounds, reference_aliases=None):
        self.reader, self.references = build_material_reader(packet, reference_aliases)
        self.model = model
        self.final_model = final_model
        self.rounds = rounds
        self.used = 0
        self.ask_user = None
        from step_user_question import ask_user_available, create_ask_user_tool
        if ask_user_available():
            self.ask_user = create_ask_user_tool("STEP_REPORTER")

    async def ainvoke(self, messages):
        from reporting.criteria import ReferencedStepReport as StepReport
        self.used += 1
        final_round = self.used >= max(1, self.rounds - 1)
        if final_round:
            from prompt_loader import load_prompt
            messages.append({"role": "user", "content": load_prompt("reporters/finalize_step_report")})
            if self.ask_user is None:
                return await self.final_model.ainvoke(messages)

        tools = [StepReport]
        if not final_round:
            tools.insert(0, self.reader)
        if self.ask_user is not None:
            tools.insert(0, self.ask_user)
        response = await self.model.bind_tools(tools).ainvoke(messages)
        calls = getattr(response, "tool_calls", [])
        ask_calls = [
            c for c in calls
            if self.ask_user is not None and c["name"] == self.ask_user.name
        ]
        if ask_calls:
            call = ask_calls[0]
            # A question is exclusive: any report generated before the human
            # answer is discarded and regenerated in the same Reporter role.
            messages.append(response.model_copy(update={"tool_calls": [call]}))
            answer = await self.ask_user.ainvoke(call.get("args") or {})
            messages.append(ToolMessage(
                content=str(answer),
                tool_call_id=call["id"],
                name=self.ask_user.name,
            ))
            return {"human_answer": True}
        final = [c for c in calls if c["name"] == "StepReport"]
        if final and len(calls) == 1:
            return {"parsed":final[0]["args"], "raw":response}
        read_calls = [c for c in calls if c["name"] == self.reader.name]
        known_calls = [
            c for c in calls
            if c["name"] in {
                self.reader.name,
                "StepReport",
                *( [self.ask_user.name] if self.ask_user is not None else [] ),
            }
        ]
        if read_calls and len(known_calls) == len(calls):
            messages.append(response)
            for call in read_calls:
                try:
                    from observability import trace_span, set_span_output
                    with trace_span("Reviewer / Read Materials", kind="tool", input_value=call["args"]) as span:
                        result = await self.reader.ainvoke(call["args"])
                        set_span_output(span, result)
                    status = "success"
                except Exception as error:
                    result = str(error)
                    status = "error"
                messages.append(ToolMessage(content=result, tool_call_id=call["id"], status=status))
            if final:
                for call in final:
                    messages.append(ToolMessage(
                        content=(
                            "StepReport deferred: this round also read materials. "
                            "Use the returned material pages, then submit StepReport alone."
                        ),
                        tool_call_id=call["id"],
                        status="error",
                    ))
                messages.append({
                    "role": "user",
                    "content": (
                        "本轮同时请求读取材料和提交StepReport。Harness已执行所有读取，"
                        "但拒绝了读取结果返回前生成的报告。请先使用上面的材料结果，"
                        "下一轮只提交一次StepReport；不要重复已成功的读取。"
                    ),
                })
            return {"material_read": True}
        return {
            "parsing_error": (
                "Only read_review_materials and StepReport are accepted here. "
                "When both are requested, Harness executes the reads and defers the report."
            )
        }
