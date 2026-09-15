"""Generate a resumable 10k SAVE/SKIP memory-gate dataset with Qwen3.7 Flash."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["PHOENIX_TRACING_ENABLED"] = "false"

from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import _build_chat_model
from config import load_settings
from cost_accounting import estimate_cost, summarize_cost

SEEDS_PATH = ROOT / "training" / "memory_write_gate" / "seeds.json"
OUTPUT_PATH = ROOT / "training" / "memory_write_gate" / "memory_write_gate_10000.jsonl"
PROMPT_PATH = ROOT / "prompts" / "training" / "memory_write_gate_generation.md"
PRIVATE_ROOT = ROOT / ".agent" / "memory-write-gate-generation"
LABEL_IDS = {"SKIP": 0, "SAVE": 1}
FOCUSES = (
    "中文口语、省略和错别字",
    "英文及自然中英混合表达",
    "中文名、英文名和人物归属",
    "邮件、工作、项目和客户场景",
    "家庭、健康、饮食和出行安排",
    "这次与以后、临时与长期的细微差别",
    "否定、转折、引用和不确定表达",
    "短句与带多个条件的自然长句",
)


def normalized(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()


def extract_usage(message: Any) -> dict[str, int]:
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


def parse_samples(content: Any) -> list[dict[str, str]]:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    payload = json.loads(text)
    rows = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("response must contain a samples list")
    parsed: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get("text", "")).strip()
        language = str(row.get("language", "")).strip().lower()
        domain = str(row.get("domain", "")).strip()[:80] or "general"
        difficulty = str(row.get("difficulty", "")).strip().lower()
        if 3 <= len(value) <= 600 and language in {"zh", "en", "mixed"} and difficulty in {"easy", "medium", "hard"}:
            parsed.append({"text": value, "language": language, "domain": domain, "difficulty": difficulty})
    return parsed


def build_model():
    settings = load_settings()
    source = settings.role_models["scheduler"]
    return _build_chat_model(
        model_provider="qwen",
        model_name="qwen3.7-flash",
        max_tokens=12000,
        timeout_seconds=180,
        thinking_enabled=False,
        api_key=source.api_key,
        base_url=source.base_url,
        max_retries=0,
        token_limit_parameter="max_completion_tokens",
        trace_role="memory_write_gate_generator",
        extra_body={"enable_thinking": False, "preserve_thinking": False},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-per-seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-calls-per-seed", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.batch_size <= 80 or not 1 <= args.concurrency <= 12:
        raise SystemExit("invalid batch size or concurrency")

    seeds = json.loads(SEEDS_PATH.read_text(encoding="utf-8-sig"))
    if len(seeds) != 100 or len({seed["seed_id"] for seed in seeds}) != 100:
        raise SystemExit("expected exactly 100 unique seed families")
    if Counter(seed["label"] for seed in seeds) != {"SAVE": 50, "SKIP": 50}:
        raise SystemExit("expected 50 SAVE and 50 SKIP seed families")
    if args.dry_run:
        print(json.dumps({
            "seeds": len(seeds),
            "target_rows": len(seeds) * args.target_per_seed,
            "labels": dict(Counter(seed["label"] for seed in seeds)),
            "splits": dict(Counter(seed["split"] for seed in seeds)),
            "categories": dict(Counter(seed["category"] for seed in seeds)),
        }, ensure_ascii=False, indent=2))
        return

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    run_dir = PRIVATE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "checkpoint.private.jsonl"
    model = build_model()
    seen: set[str] = set()
    seen_lock = threading.Lock()
    write_lock = threading.Lock()
    accounting_lock = threading.Lock()
    usage_total: Counter[str] = Counter()
    costs: list[dict[str, Any]] = []
    results: dict[str, list[dict[str, str]]] = {}
    started = time.monotonic()

    def record(payload: dict[str, Any]) -> None:
        with write_lock, checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def generate_seed(seed: dict[str, Any]):
        accepted: list[dict[str, str]] = []
        local_seen = {normalized(seed["text"])}
        calls = 0
        while len(accepted) < args.target_per_seed and calls < args.max_calls_per_seed:
            calls += 1
            needed = args.target_per_seed - len(accepted)
            request_count = min(args.batch_size, needed + 8)
            request = {
                "fixed_label": seed["label"],
                "seed_id": seed["seed_id"],
                "seed_message": seed["text"],
                "boundary_note": seed["note"],
                "category": seed["category"],
                "variation_focus": FOCUSES[(calls - 1) % len(FOCUSES)],
                "requested_count": request_count,
                "strict_count_requirement": f"必须返回恰好 {request_count} 条",
                "avoid_recent_examples": [row["text"] for row in accepted[-20:]],
                "output_schema": {"samples": [{
                    "text": "自然的用户消息",
                    "language": "zh|en|mixed",
                    "domain": "简短场景",
                    "difficulty": "easy|medium|hard",
                }]},
            }
            try:
                message = model.invoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(request, ensure_ascii=False))],
                    response_format={"type": "json_object"},
                    temperature=1.0,
                )
                usage = extract_usage(message)
                cost = estimate_cost("qwen3.7-flash", usage)
                parsed = parse_samples(message.content)
                with accounting_lock:
                    usage_total.update(usage)
                    costs.append(cost)
                added: list[dict[str, str]] = []
                for row in parsed:
                    key = normalized(row["text"])
                    if not key or key in local_seen:
                        continue
                    with seen_lock:
                        if key in seen:
                            continue
                        seen.add(key)
                    local_seen.add(key)
                    accepted.append(row)
                    added.append(row)
                    if len(accepted) == args.target_per_seed:
                        break
                record({"seed_id": seed["seed_id"], "call": calls, "requested": request_count,
                        "parsed": len(parsed), "added": len(added), "accepted_total": len(accepted),
                        "usage": usage, "cost": cost, "added_rows": added})
            except Exception as error:
                record({"seed_id": seed["seed_id"], "call": calls,
                        "error_type": type(error).__name__, "error": str(error)[:1000]})
        if len(accepted) != args.target_per_seed:
            raise RuntimeError(f"{seed['seed_id']} produced {len(accepted)}/{args.target_per_seed}")
        (run_dir / f"{seed['seed_id']}.private.json").write_text(
            json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
        return seed["seed_id"], accepted, calls

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(generate_seed, seed): seed for seed in seeds}
        for completed, future in enumerate(as_completed(futures), 1):
            seed_id, accepted, calls = future.result()
            results[seed_id] = accepted
            print(f"[{completed:03d}/100] {seed_id}: {len(accepted)} rows in {calls} calls", flush=True)

    final_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for variant, row in enumerate(results[seed["seed_id"]], 1):
            final_rows.append({
                "id": f"{seed['seed_id']}-{variant:03d}", "text": row["text"],
                "label": seed["label"], "label_id": LABEL_IDS[seed["label"]],
                "split": seed["split"], "seed_id": seed["seed_id"],
                "category": seed["category"], "language": row["language"],
                "domain": row["domain"], "difficulty": row["difficulty"],
            })
    expected = len(seeds) * args.target_per_seed
    if len(final_rows) != expected or len({normalized(row["text"]) for row in final_rows}) != expected:
        raise RuntimeError("final count or uniqueness check failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "qwen3.7-flash", "thinking_enabled": False,
        "generation_mode": f"{args.concurrency}-way concurrent bounded batches",
        "rows": len(final_rows), "seed_families": len(seeds),
        "labels": dict(Counter(row["label"] for row in final_rows)),
        "splits": dict(Counter(row["split"] for row in final_rows)),
        "languages": dict(Counter(row["language"] for row in final_rows)),
        "categories": dict(Counter(row["category"] for row in final_rows)),
        "usage": dict(usage_total), "cost": summarize_cost(costs),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output": str(args.output.relative_to(ROOT)),
        "private_checkpoint": str(checkpoint.relative_to(ROOT)),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output.parent / "generation_manifest.json").write_text(
        json.dumps({k: v for k, v in manifest.items() if k != "private_checkpoint"}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
