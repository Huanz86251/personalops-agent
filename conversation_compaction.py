"""Rolling JSON summaries of user-visible dialogue; never execution traces."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import tiktoken

from prompt_loader import render_prompt
from structured_output_policy import textual_schema_fallback
from schemas.compaction import ConversationSummary, SummaryCheckpoint
from compaction_runtime import arun_compaction


@lru_cache(maxsize=1)
def _encoding():
    # A local, reproducible estimate, NOT Qwen's exact tokenizer or billing count.
    return tiktoken.get_encoding("cl100k_base")


def dialogue_token_estimate(turns: list[list[Any]]) -> int:
    """Count dialogue text and role framing, preserving CJK and special strings."""
    return sum(
        len(_encoding().encode(message.content, disallowed_special=())) + 4
        for turn in turns for message in turn
    )


def read_summary_checkpoint(raw: Any, total_turns: int) -> SummaryCheckpoint | None:
    """Legacy prose/count checkpoints are rebuilt from canonical raw messages."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        checkpoint = SummaryCheckpoint.model_validate_json(raw)
    except ValueError:
        return None
    if not 1 <= checkpoint.covered_turns <= total_turns:
        return None
    return checkpoint


async def compress_dialogue(
    model: Any,
    *,
    previous: ConversationSummary | None,
    turns: list[list[Any]],
    first_turn: int,
    max_chars: int,
) -> ConversationSummary:
    """Generate validated JSON; failure must not consume unsummarized history."""
    source = {
        "previous_summary": previous.model_dump() if previous is not None else None,
        "new_dialogue": [
            {"turn": first_turn + index,
             "messages": [{"role": item.role, "content": item.content} for item in turn]}
            for index, turn in enumerate(turns)
        ],
    }
    schema_guidance = textual_schema_fallback(model, ConversationSummary, heading="\nJSON Schema:\n")
    messages = [
        {"role": "system", "content": render_prompt("conversation/summary")
         + f"\n输出 JSON 总长度不得超过 {max_chars} 个字符。"
         + schema_guidance},
        {"role": "user", "content": json.dumps(source, ensure_ascii=False)},
    ]
    summary, _ = await arun_compaction(
        model, ConversationSummary, messages,
        core_fields=("user_requirements", "decisions", "assistant_results", "pending_items"),
        max_chars=max_chars,
    )
    return summary
