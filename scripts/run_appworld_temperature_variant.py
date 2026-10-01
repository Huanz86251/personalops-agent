"""Run a matched four-way Test-N low-temperature repeat after baseline.

The baseline is never modified or retried. This script applies process-local
environment overrides to the second batch, preserving existing extra_body
fields and leaving the repository .env untouched.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dotenv import dotenv_values

from complete_appworld_full_batch import _complete_half, _read, _status, _zip_traces


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASELINE_ID = "test-normal-full-p4-20260916-demo"
VARIANT_ID = "test-normal-full-p4-lowtemp-20260916-demo"
COMPARISON_ID = "test-normal-lowtemp-20260916"
GLM_TEMPERATURE = 0.5
QWEN38_THINKING_TEMPERATURE = 0.6
GLM_ROLES = ("general", "code", "code_reviewer", "scheduler")
PROJECT = "AppWorld Test-N Low Temp 20260916 Demo p4"
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], log: Path, env: dict[str, str]) -> None:
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            check=False, creationflags=CREATE_NO_WINDOW,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}; inspect {log}")


def _build_variant_environment() -> tuple[dict[str, str], dict[str, object]]:
    from config import load_settings
    from model_roles import load_role_models

    settings = load_settings()
    role_configs = load_role_models(settings)
    local_values = dotenv_values(ROOT / ".env")
    environment = os.environ.copy()
    effective: dict[str, object] = {}
    targets = {role: GLM_TEMPERATURE for role in GLM_ROLES}
    targets.update({
        role: QWEN38_THINKING_TEMPERATURE
        for role, config in role_configs.items()
        if config.model == "qwen3.8-flash" and config.thinking_enabled
    })
    if not any(value == QWEN38_THINKING_TEMPERATURE for value in targets.values()):
        raise RuntimeError("No Qwen3.8 thinking roles found")
    for role, temperature in targets.items():
        config = role_configs[role]
        if role in GLM_ROLES and config.model != "ZHIPU/GLM-5.3-Flash":
            raise RuntimeError(f"{role} no longer routes to the intended GLM model")
        if role not in GLM_ROLES and (config.model != "qwen3.8-flash" or not config.thinking_enabled):
            raise RuntimeError(f"{role} is not a Qwen3.8 thinking role")
        name = f"{role.upper()}_LLM_EXTRA_BODY_JSON"
        raw = environment.get(name, local_values.get(name) or "{}")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{name} is not a JSON object")
        before_keys = sorted(parsed)
        if "temperature" in parsed:
            raise RuntimeError(f"{name} already sets temperature; baseline would be ambiguous")
        parsed["temperature"] = temperature
        environment[name] = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        effective[role] = {
            "provider": config.provider,
            "model": config.model,
            "temperature": temperature,
            "thinking_enabled_at_role_level": config.thinking_enabled,
            "other_extra_body_keys_preserved": before_keys,
        }
    return environment, effective


def _paired_comparison(baseline: Path, variant: Path) -> dict[str, object]:
    base_metrics = _read(baseline / "full-aggregate.private.json")
    variant_metrics = _read(variant / "full-aggregate.private.json")
    base_index = _read(baseline / "demo-trace-index.private.json")
    variant_index = _read(variant / "demo-trace-index.private.json")
    if not all(isinstance(value, dict) for value in (base_metrics, variant_metrics, base_index, variant_index)):
        raise RuntimeError("Both full aggregates and trace indexes are required")
    base_outcomes = {entry["task_id"]: entry["outcome"] for entry in base_index["traces"]}
    variant_outcomes = {entry["task_id"]: entry["outcome"] for entry in variant_index["traces"]}
    if set(base_outcomes) != set(variant_outcomes) or len(base_outcomes) != 168:
        raise RuntimeError("The two runs did not cover the same 168 unique tasks")
    pairs = Counter((base_outcomes[task], variant_outcomes[task]) for task in base_outcomes)
    return {
        "baseline_batch_id": BASELINE_ID,
        "variant_batch_id": VARIANT_ID,
        "temperature_change": "GLM roles to 0.5; Qwen3.8 thinking roles to 0.6; other roles unchanged",
        "task_count": 168,
        "paired_outcomes": {
            "both_passed": sum(n for (before, after), n in pairs.items() if before == after == "PASSED"),
            "baseline_only_passed": sum(n for (before, after), n in pairs.items() if before == "PASSED" and after != "PASSED"),
            "variant_only_passed": sum(n for (before, after), n in pairs.items() if before != "PASSED" and after == "PASSED"),
            "neither_passed": sum(n for (before, after), n in pairs.items() if before != "PASSED" and after != "PASSED"),
            "detailed_counts": {f"{before}->{after}": n for (before, after), n in sorted(pairs.items())},
        },
        "headline_metrics": {
            name: {
                "official_passed": metric["official_outcomes"]["PASSED"],
                "official_failed": metric["official_outcomes"]["FAILED"],
                "unscored": metric["official_outcomes"]["UNSCORED"],
                "conservative_success_rate": metric["official_outcomes"]["conservative_success_rate"],
                "end_to_end_mean_seconds": metric["task_end_to_end_seconds"]["mean"],
                "end_to_end_p50_seconds": metric["task_end_to_end_seconds"]["p50"],
                "end_to_end_p90_seconds": metric["task_end_to_end_seconds"]["p90"],
                "end_to_end_p95_seconds": metric["task_end_to_end_seconds"]["p95"],
                "known_total_cost_cny": metric["cost_cny"]["known_total"],
                "cost_missing_requests": metric["cost_cny"]["missing_request_count"],
                "weighted_cache_hit_ratio": metric["tokens"]["cache_hit_ratio_weighted"],
                "model_calls_total": metric["model_calls"]["total"],
            }
            for name, metric in (("baseline", base_metrics), ("low_temperature", variant_metrics))
        },
        "important_caveat": "Two stochastic runs support an observed comparison, not a causal guarantee that temperature alone explains every task outcome.",
    }


def main() -> None:
    baseline = ROOT / ".agent" / "appworld-test-normal" / BASELINE_ID
    variant = ROOT / ".agent" / "appworld-test-normal" / VARIANT_ID
    comparison = ROOT / ".agent" / "appworld-comparisons" / COMPARISON_ID
    if comparison.exists():
        previous = _read(comparison / "completion-status.private.json") or {}
        if previous.get("state") not in {"ERROR", "WAITING_FOR_BASELINE"}:
            raise RuntimeError("Existing comparison has progressed past the safe restart point")
    comparison.mkdir(parents=True, exist_ok=True)
    baseline_manifest = _read(baseline / "manifest.private.json")
    if not isinstance(baseline_manifest, dict) or baseline_manifest.get("task_count") != 168:
        raise RuntimeError("Baseline frozen manifest is missing or incomplete")
    config_files = [ROOT / ".env", ROOT / "config" / "model_thinking.json"]
    config_hashes = {str(path.relative_to(ROOT)): _digest(path) for path in config_files}
    _status(comparison, "WAITING_FOR_BASELINE", "GLM 0.5 and Qwen3.8 thinking 0.6 will start only after baseline FINISHED")
    deadline = time.monotonic() + 36 * 3600
    while True:
        state = _read(baseline / "completion-status.private.json") or {}
        if state.get("state") == "FINISHED":
            break
        if state.get("state") == "ERROR":
            raise RuntimeError("Baseline completion helper failed; second run was not started")
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for baseline completion")
        time.sleep(30)
    if not (baseline / "full-aggregate.private.json").is_file():
        raise RuntimeError("Baseline aggregate missing despite FINISHED status")
    if any(_digest(ROOT / relative) != digest for relative, digest in config_hashes.items()):
        raise RuntimeError("Model configuration changed during baseline; refusing incomparable repeat")
    if variant.exists():
        raise RuntimeError("Temperature-0.5 batch directory already exists; refusing overwrite")
    environment, effective = _build_variant_environment()
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_batch_id": BASELINE_ID,
        "variant_batch_id": VARIANT_ID,
        "seed": baseline_manifest["seed"],
        "image": baseline_manifest["image"],
        "source_digest": baseline_manifest["source_digest"],
        "baseline_config_sha256": config_hashes,
        "temperature_overrides": effective,
        "other_roles": "unchanged, including Qwen3.7 thinking and Qwen3.8 non-thinking selector",
        "override_scope": "process-local; repository .env is not edited",
        "parallelism": 4,
        "phoenix_project": PROJECT,
    }
    (comparison / "comparison-plan.private.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _status(comparison, "PREPARING_VARIANT", "Freezing the same 168-task selection")
    command = [
        sys.executable, "-X", "utf8", "-m", "evals.appworld.run_test_normal_batch",
        "--batch-id", VARIANT_ID, "--seed", str(baseline_manifest["seed"]),
        "--image", baseline_manifest["image"],
    ]
    _run([*command, "--prepare"], comparison / "prepare.log", environment)
    variant_manifest = _read(variant / "manifest.private.json")
    if not isinstance(variant_manifest, dict):
        raise RuntimeError("Variant manifest missing after prepare")
    for half in ("A", "B"):
        if variant_manifest["halves"][half]["selection_sha256"] != baseline_manifest["halves"][half]["selection_sha256"]:
            raise RuntimeError(f"Half {half} task selection differs from baseline")
    if variant_manifest["source_digest"] != baseline_manifest["source_digest"]:
        raise RuntimeError("Agent source changed between baseline and variant")
    (variant / "experiment-config.private.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for half in ("A", "B"):
        _status(comparison, f"RUNNING_VARIANT_{half}", "Four-way paid evaluation with GLM 0.5 and Qwen3.8 thinking 0.6")
        _run(
            [*command, "--half", half, "--parallelism", "4", "--phoenix-project", PROJECT, "--allow-paid"],
            comparison / f"half-{half.lower()}-launcher.log", environment,
        )
        if not _complete_half(variant, half, int(variant_manifest["halves"][half]["count"])):
            raise RuntimeError(f"Variant half {half} ledger is incomplete")
    _status(comparison, "AGGREGATING_VARIANT", "Computing official metrics and matched comparison")
    _run(
        [sys.executable, "-X", "utf8", "scripts/appworld_full_batch_stats.py", "--batch-id", VARIANT_ID],
        comparison / "aggregate-launcher.log", environment,
    )
    files, size = _zip_traces(variant)
    matched = _paired_comparison(baseline, variant)
    (comparison / "paired-comparison.private.json").write_text(
        json.dumps(matched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _status(comparison, "FINISHED", f"Both 168-task runs complete; variant trace archive has {files} files ({size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        comparison_path = ROOT / ".agent" / "appworld-comparisons" / COMPARISON_ID
        if comparison_path.is_dir():
            _status(comparison_path, "ERROR", f"{type(exc).__name__}: {exc}")
        raise
