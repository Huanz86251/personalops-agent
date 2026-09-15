"""Add 10k long-form families to the memory write-gate dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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

from cost_accounting import estimate_cost, summarize_cost
from generate_memory_write_gate_dataset import build_model, extract_usage, normalized

SOURCE = ROOT / "training" / "memory_write_gate" / "memory_write_gate_10000.jsonl"
OUTPUT = ROOT / "training" / "memory_write_gate" / "memory_write_gate_20000.jsonl"
SEEDS = ROOT / "training" / "memory_write_gate" / "seeds_long.json"
PROMPT = ROOT / "prompts" / "training" / "memory_write_gate_long_generation.md"
PRIVATE_ROOT = ROOT / ".agent" / "memory-write-gate-generation"
LABEL_IDS = {"SKIP": 0, "SAVE": 1}
FOCUSES = (
    "人物关系与中英文姓名",
    "邮件、工作和客户场景",
    "一次性请求中夹带长期要求",
    "否定、转折和时间范围",
    "健康、家庭和出行",
    "项目约束、决定和任务状态",
)


def parse_samples(content: Any, expected_targets: list[int]) -> list[dict[str, Any]]:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    payload = json.loads(text)
    rows = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("response must contain a samples list")
    parsed = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        value = str(row.get("text", "")).strip()
        language = str(row.get("language", "")).strip().lower()
        domain = str(row.get("domain", "")).strip()[:80] or "general"
        difficulty = str(row.get("difficulty", "")).strip().lower()
        requested_target = expected_targets[min(index, len(expected_targets) - 1)]
        target_chars = requested_target
        actual_chars = len(value)
        if language not in {"zh", "en", "mixed"}:
            has_cjk = any("\u4e00" <= char <= "\u9fff" for char in value)
            has_latin = any(char.isascii() and char.isalpha() for char in value)
            language = "mixed" if has_cjk and has_latin else ("zh" if has_cjk else "en")
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"
        if 50 <= actual_chars <= 320:
            parsed.append({"text": value, "language": language, "domain": domain,
                           "difficulty": difficulty, "target_chars": target_chars,
                           "actual_chars": actual_chars})
    return parsed


def rng_for(seed_id: str, call: int) -> random.Random:
    digest = hashlib.sha256(f"20260915|{seed_id}|{call}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-per-seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-calls-per-seed", type=int, default=20)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = json.loads(SEEDS.read_text(encoding="utf-8-sig"))
    if len(seeds) != 100 or Counter(seed["label"] for seed in seeds) != {"SAVE": 50, "SKIP": 50}:
        raise SystemExit("expected 50 SAVE and 50 SKIP long seed families")
    existing = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(existing) != 10000:
        raise SystemExit("expected the frozen 10k source dataset")
    if args.dry_run:
        print(json.dumps({"existing": len(existing), "new_seeds": len(seeds),
                          "new_rows": len(seeds) * args.target_per_seed,
                          "final_rows": len(existing) + len(seeds) * args.target_per_seed,
                          "labels": dict(Counter(seed["label"] for seed in seeds)),
                          "splits": dict(Counter(seed["split"] for seed in seeds)),
                          "target_chars": [50, 250]}, ensure_ascii=False, indent=2))
        return

    system_prompt = PROMPT.read_text(encoding="utf-8").strip()
    run_dir = PRIVATE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-long10k")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "checkpoint.private.jsonl"
    model = build_model()
    seen = {normalized(row["text"]) for row in existing}
    seen_lock = threading.Lock()
    write_lock = threading.Lock()
    accounting_lock = threading.Lock()
    usage_total: Counter[str] = Counter()
    costs: list[dict[str, Any]] = []
    results: dict[str, list[dict[str, Any]]] = {}
    started = time.monotonic()

    def record(payload):
        with write_lock, checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def generate_seed(seed):
        accepted = []
        local_seen = {normalized(seed["text"])}
        calls = 0
        while len(accepted) < args.target_per_seed and calls < args.max_calls_per_seed:
            calls += 1
            needed = args.target_per_seed - len(accepted)
            request_count = min(args.batch_size, needed + 10)
            target_rng = rng_for(seed["seed_id"], calls)
            targets = [target_rng.randint(50, 250) for _ in range(request_count)]
            request = {
                "fixed_label": seed["label"], "seed_id": seed["seed_id"],
                "seed_message": seed["text"], "boundary_note": seed["note"],
                "category": seed["category"], "variation_focus": FOCUSES[(calls - 1) % len(FOCUSES)],
                "target_chars_in_output_order": targets,
                "requested_count": request_count,
                "strict_count_requirement": f"必须返回恰好{request_count}条，并逐项原样带回target_chars",
                "avoid_recent_examples": [row["text"] for row in accepted[-16:]],
                "output_schema": {"samples": [{"target_chars": 137, "text": "自然用户消息",
                    "language": "zh|en|mixed", "domain": "简短场景", "difficulty": "easy|medium|hard"}]},
            }
            try:
                message = model.invoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(request, ensure_ascii=False))],
                    response_format={"type": "json_object"}, temperature=1.0)
                usage = extract_usage(message)
                cost = estimate_cost("qwen3.7-flash", usage)
                parsed = parse_samples(message.content, targets)
                with accounting_lock:
                    usage_total.update(usage); costs.append(cost)
                added = []
                for row in parsed:
                    key = normalized(row["text"])
                    if not key or key in local_seen:
                        continue
                    with seen_lock:
                        if key in seen:
                            continue
                        seen.add(key)
                    local_seen.add(key); accepted.append(row); added.append(row)
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

    new_rows = []
    for seed in seeds:
        for variant, row in enumerate(results[seed["seed_id"]], 1):
            new_rows.append({"id": f"{seed['seed_id']}-{variant:03d}", "text": row["text"],
                "label": seed["label"], "label_id": LABEL_IDS[seed["label"]],
                "split": seed["split"], "seed_id": seed["seed_id"], "category": seed["category"],
                "language": row["language"], "domain": row["domain"], "difficulty": row["difficulty"],
                "target_chars": row["target_chars"], "actual_chars": row["actual_chars"]})
    final_rows = [*existing, *new_rows]
    if len(new_rows) != 10000 or len(final_rows) != 20000 or len({normalized(r["text"]) for r in final_rows}) != 20000:
        raise RuntimeError("final count or uniqueness check failed")
    with args.output.open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    previous = json.loads((SOURCE.parent / "generation_manifest.json").read_text(encoding="utf-8"))
    new_cost = summarize_cost(costs)
    lengths = sorted(row["actual_chars"] for row in new_rows)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "qwen3.7-flash", "thinking_enabled": False,
        "rows": len(final_rows), "existing_rows": len(existing), "new_rows": len(new_rows),
        "seed_families": 200, "new_seed_families": len(seeds),
        "labels": dict(Counter(row["label"] for row in final_rows)),
        "splits": dict(Counter(row["split"] for row in final_rows)),
        "languages": dict(Counter(row["language"] for row in final_rows)),
        "new_length_chars": {"min": lengths[0], "p50": lengths[len(lengths)//2],
                             "p95": lengths[int(len(lengths)*0.95)-1], "max": lengths[-1]},
        "new_usage": dict(usage_total), "new_cost": new_cost,
        "combined_known_cost_cny": float(previous["cost"]["known_total_cny"]) + float(new_cost["known_total_cny"]),
        "elapsed_seconds": round(time.monotonic()-started, 3), "output": str(args.output.relative_to(ROOT)),
        "private_checkpoint": str(checkpoint.relative_to(ROOT))}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output.parent / "generation_manifest_20000.json").write_text(
        json.dumps({k:v for k,v in manifest.items() if k!="private_checkpoint"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
