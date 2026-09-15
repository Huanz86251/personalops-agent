"""Generate a balanced synthetic intent-routing dataset with Qwen3.7 Flash."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import _build_chat_model
from config import load_settings
from cost_accounting import estimate_cost, summarize_cost
SEEDS_PATH = ROOT / "training" / "scope_intent" / "seeds.json"
OUTPUT_PATH = ROOT / "training" / "scope_intent" / "scope_intent_4000.jsonl"
PRIVATE_ROOT = ROOT / ".agent" / "scope-intent-generation"
LABEL_IDS = {"DIRECT_RESPONSE": 0, "REQUIRES_SCOPE_CONTRACT": 1}

SYSTEM_PROMPT = """你是合成分类训练数据的编辑器。只生成用户请求，不回答请求。最终只返回符合给定结构的 JSON 对象。
Harness 已经固定本批标签；你不得重新判断或改变标签。

DIRECT_RESPONSE：当前消息中的文字已经足够，助手可以直接回答；不得读取文件、账户、网页、数据库或运行环境，也不执行真实操作。询问“如何做”可以属于此类。
REQUIRES_SCOPE_CONTRACT：要完成请求必须读取或改变外部状态，或调用工具、操作文件、账户、网页、数据库、代码环境。即使操作很简单也属于此类。

生成自然、真实、彼此差异明显的请求。覆盖中文和英文、正式和口语、短句和带条件的长句。保持种子的核心边界，但更换对象、行业、表达和复杂度。不要出现标签名、解释、答案、序号、Markdown。"""


def _normalized(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def _extract_usage(message: Any) -> dict[str, int]:
    metadata = dict(getattr(message, "response_metadata", None) or {})
    usage = metadata.get("token_usage") or metadata.get("usage") or {}
    usage_meta = dict(getattr(message, "usage_metadata", None) or {})
    details = usage_meta.get("input_token_details") or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", usage_meta.get("input_tokens", 0)) or 0),
        "output_tokens": int(usage.get("completion_tokens", usage_meta.get("output_tokens", 0)) or 0),
        "cache_read_tokens": int(
            usage.get("prompt_tokens_details", {}).get("cached_tokens", details.get("cache_read", 0)) or 0
        ),
    }


def _parse_samples(content: Any) -> list[dict[str, str]]:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    payload = json.loads(text)
    rows = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("response must contain a samples list")
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get("text", "")).strip()
        language = str(row.get("language", "")).strip().lower()
        domain = str(row.get("domain", "")).strip()[:80]
        difficulty = str(row.get("difficulty", "")).strip().lower()
        if 4 <= len(value) <= 500 and language in {"zh", "en"} and difficulty in {"easy", "medium", "hard"}:
            parsed.append({"text": value, "language": language, "domain": domain or "general", "difficulty": difficulty})
    return parsed


def _model():
    settings = load_settings()
    source = settings.role_models["scheduler"]
    return _build_chat_model(
        model_provider="qwen", model_name="qwen3.7-flash", max_tokens=16000,
        timeout_seconds=180, thinking_enabled=False, api_key=source.api_key,
        base_url=source.base_url, max_retries=0,
        token_limit_parameter="max_completion_tokens", trace_role="scope_intent_generator",
        extra_body={"enable_thinking": False, "preserve_thinking": False},
    )


def main() -> None:
    import os
    os.environ["PHOENIX_TRACING_ENABLED"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-seed", type=int, default=100)
    parser.add_argument("--max-calls-per-seed", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.per_seed <= 150:
        raise SystemExit("--per-seed must be between 1 and 150")

    seeds = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))
    for seed in seeds:
        if seed["label"] not in LABEL_IDS or seed["split"] not in {"train", "validation", "test"}:
            raise SystemExit(f"invalid seed: {seed}")
    if args.dry_run:
        print(json.dumps({"seeds": len(seeds), "target": len(seeds) * args.per_seed,
                          "labels": Counter(x["label"] for x in seeds),
                          "splits": Counter(x["split"] for x in seeds)}, ensure_ascii=False, default=dict))
        return

    run_dir = PRIVATE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "checkpoint.private.jsonl"
    model = _model()
    all_rows: list[dict[str, Any]] = []
    global_seen: set[str] = set()
    costs: list[dict[str, Any]] = []
    usage_total = Counter()
    started = time.monotonic()

    for index, seed in enumerate(seeds, 1):
        accepted: list[dict[str, str]] = []
        seed_seen = {_normalized(seed["text"])}
        for attempt in range(1, args.max_calls_per_seed + 1):
            needed = args.per_seed - len(accepted)
            if needed <= 0:
                break
            request_count = min(120, max(needed + 10, 20))
            focuses = ["行业与对象", "口语与省略表达", "复杂限定条件", "中英文场景", "边界和反例措辞"]
            focus = focuses[(attempt - 1) % len(focuses)]
            prompt = {
                "fixed_label": seed["label"], "seed_id": seed["seed_id"],
                "seed_request": seed["text"], "boundary_note": seed["note"],
                "requested_count": request_count,
                "strict_count_requirement": f"必须返回恰好 {request_count} 条；少于该数量视为失败",
                "variation_focus": focus,
                "avoid_recent_examples": [row["text"] for row in accepted[-30:]],
                "output_schema": {"samples": [{"text": "用户请求", "language": "zh|en",
                                                  "domain": "简短领域", "difficulty": "easy|medium|hard"}]},
            }
            message = model.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
            ], response_format={"type": "json_object"}, temperature=0.9)
            usage = _extract_usage(message)
            usage_total.update(usage)
            cost = estimate_cost("qwen3.7-flash", usage)
            costs.append(cost)
            parsed = _parse_samples(message.content)
            added = 0
            for row in parsed:
                key = _normalized(row["text"])
                if not key or key in seed_seen or key in global_seen:
                    continue
                seed_seen.add(key)
                global_seen.add(key)
                accepted.append(row)
                added += 1
                if len(accepted) == args.per_seed:
                    break
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"seed_id": seed["seed_id"], "attempt": attempt,
                                         "requested": request_count, "parsed": len(parsed),
                                         "added": added, "accepted_total": len(accepted),
                                         "usage": usage, "cost": cost,
                                         "accepted_rows": accepted}, ensure_ascii=False) + "\n")
        if len(accepted) != args.per_seed:
            raise RuntimeError(f"{seed['seed_id']} produced only {len(accepted)}/{args.per_seed} unique samples")
        (run_dir / f"{seed['seed_id']}.private.json").write_text(
            json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
        for variant, row in enumerate(accepted, 1):
            all_rows.append({
                "id": f"{seed['seed_id']}-{variant:03d}", "text": row["text"],
                "label": seed["label"], "label_id": LABEL_IDS[seed["label"]],
                "split": seed["split"], "seed_id": seed["seed_id"],
                "language": row["language"], "domain": row["domain"],
                "difficulty": row["difficulty"],
            })
        print(f"[{index:02d}/{len(seeds)}] {seed['seed_id']} {seed['label']}: {len(accepted)}")

    expected = len(seeds) * args.per_seed
    if len(all_rows) != expected or len(global_seen) != expected:
        raise RuntimeError("final dataset count or uniqueness check failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "model": "qwen3.7-flash",
        "thinking_enabled": False, "rows": len(all_rows),
        "labels": dict(Counter(row["label"] for row in all_rows)),
        "splits": dict(Counter(row["split"] for row in all_rows)),
        "languages": dict(Counter(row["language"] for row in all_rows)),
        "usage": dict(usage_total), "cost": summarize_cost(costs),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output": str(args.output.relative_to(ROOT)), "private_checkpoint": str(checkpoint.relative_to(ROOT)),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output.parent / "generation_manifest.json").write_text(
        json.dumps({k: v for k, v in manifest.items() if k != "private_checkpoint"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
