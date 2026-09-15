"""Expand the scope-intent dataset to 20k with concurrent resumable Qwen batches."""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ["PHOENIX_TRACING_ENABLED"] = "false"

from langchain_core.messages import HumanMessage, SystemMessage
from generate_scope_intent_dataset import (
    ROOT, SEEDS_PATH, LABEL_IDS, SYSTEM_PROMPT, _extract_usage,
    _model, _normalized, _parse_samples,
)
from cost_accounting import estimate_cost, summarize_cost

SOURCE = ROOT / "training" / "scope_intent" / "scope_intent_4000.jsonl"
OUTPUT = ROOT / "training" / "scope_intent" / "scope_intent_20000.jsonl"
PRIVATE_ROOT = ROOT / ".agent" / "scope-intent-generation"
FOCUSES = [
    "更换行业和对象", "口语、省略和错别字", "主语与条件归属",
    "并集、交集、差集和排除条件", "咨询做法与要求真实执行的边界",
    "跨系统交接与动态状态", "写入、回读和验收", "中英文混合与长短句",
]


def compact(row):
    return {key: row[key] for key in ("text", "language", "domain", "difficulty")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-per-seed", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-calls-per-seed", type=int, default=16)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 10 <= args.batch_size <= 100 or not 1 <= args.concurrency <= 12:
        raise SystemExit("invalid batch size or concurrency")

    seeds = json.loads(SEEDS_PATH.read_text(encoding="utf-8-sig"))
    existing_rows = [
        json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing_by_seed = {}
    for row in existing_rows:
        existing_by_seed.setdefault(row["seed_id"], []).append(compact(row))

    run_dir = PRIVATE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-20k")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "checkpoint.private.jsonl"
    model = _model()
    global_seen = {_normalized(row["text"]) for row in existing_rows}
    seen_lock = threading.Lock()
    write_lock = threading.Lock()
    accounting_lock = threading.Lock()
    costs = []
    usage_total = Counter()
    results = {}
    started = time.monotonic()

    def record(payload):
        with write_lock:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def generate_seed(seed):
        accepted = list(existing_by_seed.get(seed["seed_id"], []))
        local_seen = {_normalized(row["text"]) for row in accepted}
        calls = 0
        while len(accepted) < args.target_per_seed and calls < args.max_calls_per_seed:
            calls += 1
            needed = args.target_per_seed - len(accepted)
            request_count = min(args.batch_size, needed + 8)
            prompt = {
                "fixed_label": seed["label"], "seed_id": seed["seed_id"],
                "seed_request": seed["text"], "boundary_note": seed["note"],
                "requested_count": request_count,
                "strict_count_requirement": f"必须返回恰好 {request_count} 条；少于该数量视为失败",
                "variation_focus": FOCUSES[(calls - 1) % len(FOCUSES)],
                "avoid_recent_examples": [row["text"] for row in accepted[-24:]],
                "output_schema": {"samples": [{"text": "用户请求", "language": "zh|en",
                    "domain": "简短领域", "difficulty": "easy|medium|hard"}]},
            }
            try:
                message = model.invoke(
                    [SystemMessage(content=SYSTEM_PROMPT),
                     HumanMessage(content=json.dumps(prompt, ensure_ascii=False))],
                    response_format={"type": "json_object"}, temperature=1.0,
                )
                usage = _extract_usage(message)
                cost = estimate_cost("qwen3.7-flash", usage)
                parsed = _parse_samples(message.content)
                with accounting_lock:
                    usage_total.update(usage)
                    costs.append(cost)
                added = []
                for row in parsed:
                    key = _normalized(row["text"])
                    if not key or key in local_seen:
                        continue
                    with seen_lock:
                        if key in global_seen:
                            continue
                        global_seen.add(key)
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
            raise RuntimeError(
                f"{seed['seed_id']} produced {len(accepted)}/{args.target_per_seed}"
            )
        (run_dir / f"{seed['seed_id']}.private.json").write_text(
            json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return seed["seed_id"], accepted, calls

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(generate_seed, seed): seed for seed in seeds}
        completed = 0
        for future in as_completed(futures):
            seed_id, accepted, calls = future.result()
            results[seed_id] = accepted
            completed += 1
            print(f"[{completed:02d}/{len(seeds)}] {seed_id}: {len(accepted)} ({calls} calls)", flush=True)

    final_rows = []
    for seed in seeds:
        accepted = results[seed["seed_id"]]
        for variant, row in enumerate(accepted, 1):
            final_rows.append({
                "id": f"{seed['seed_id']}-{variant:03d}", "text": row["text"],
                "label": seed["label"], "label_id": LABEL_IDS[seed["label"]],
                "split": seed["split"], "seed_id": seed["seed_id"],
                "language": row["language"], "domain": row["domain"],
                "difficulty": row["difficulty"],
            })
    expected = len(seeds) * args.target_per_seed
    if len(final_rows) != expected:
        raise RuntimeError(f"expected {expected}, got {len(final_rows)}")
    normalized = [_normalized(row["text"]) for row in final_rows]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("duplicate text detected in final dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    previous_manifest = json.loads(
        (SOURCE.parent / "generation_manifest.json").read_text(encoding="utf-8")
    )
    expansion_cost = summarize_cost(costs)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "qwen3.7-flash", "thinking_enabled": False,
        "generation_mode": "8-way concurrent bounded batches",
        "rows": len(final_rows), "reused_rows": len(existing_rows),
        "new_rows": len(final_rows) - len(existing_rows),
        "labels": dict(Counter(row["label"] for row in final_rows)),
        "splits": dict(Counter(row["split"] for row in final_rows)),
        "languages": dict(Counter(row["language"] for row in final_rows)),
        "new_usage": dict(usage_total), "new_cost": expansion_cost,
        "prior_4000_cost": previous_manifest["cost"],
        "combined_known_cost_cny": (
            float(previous_manifest["cost"]["known_total_cny"])
            + float(expansion_cost["known_total_cny"])
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output": str(args.output.relative_to(ROOT)),
        "private_checkpoint": str(checkpoint.relative_to(ROOT)),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output.parent / "generation_manifest_20000.json").write_text(
        json.dumps({k: v for k, v in manifest.items() if k != "private_checkpoint"},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
