"""Worker-authored knowledge and Harness-owned citation receipts."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema


class HandoffKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_refs: list[str] = Field(default_factory=list, max_length=24,
        description="引用实际支持本条交接的已登记E/W工具结果，可填多个。事实、API文档和执行结果请引用；纯建议或无工具依据时填[]并说明未确认。不要猜编号。")
    evidence_check: SkipJsonSchema[Literal["RESOLVED", "PARTIAL", "UNSUPPORTED"]] = Field(default="UNSUPPORTED")
    unresolved_evidence_refs: SkipJsonSchema[list[str]] = Field(default_factory=list)
    evidence_errors: SkipJsonSchema[dict[str, str]] = Field(default_factory=dict)
    resolved_evidence: SkipJsonSchema[list[dict]] = Field(default_factory=list)

    topic: str = Field(min_length=1, max_length=300, description="Specific capability or discovery needed by the next task, not a generic progress summary.")
    source: str = Field(min_length=1, max_length=1200, description="Retrievable documentation locator: app/API name, URL, or shared artifact path/section. Do not use a private temporary path, a local short evidence ID (put it in evidence_refs instead), or phrases such as see my previous login result as a locator; the next agent cannot see it.")
    usage: str = Field(min_length=1, max_length=6000, description="Self-contained instructions based on observed docs/results. For an API include exact documented name/signature, purpose, required keyword arguments, authentication and pagination when applicable, plus a minimal call example. Explain placeholders and how to obtain their real values via documented calls; never refer to a previous local variable or copy passwords/tokens. For code/web findings include relevant command/file/URL and how to use it. Do not invent missing details: state what is still unknown.")
    observed_result: str = Field(min_length=1, max_length=3000, description="Observed return type/field names and their meaning, or verified behavior/results; distinguish documentation-only knowledge from an actually executed successful call. Preserve structure/newlines. Never include credentials or unrelated personal records.")
    next_action: str = Field(min_length=1, max_length=2000, description="What the next agent can do with this knowledge, prerequisites it must obtain, and unresolved limits. State whether prior writes already succeeded so they are not repeated.")
