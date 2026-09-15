"""Public-list-price estimates; never provider billing or historical backfills."""
import json
import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "config" / "pricing.json"

@lru_cache(maxsize=1)
def load_pricing():
    return json.loads(Path(os.getenv("MODEL_PRICING_PATH", str(DEFAULT_PATH))).read_text(encoding="utf-8"))

def estimate_cost(model, usage, config=None):
    result = {"status": "unknown", "currency": "CNY", "model": model}
    try:
        config = config if config is not None else load_pricing()
        if config.get("currency") != "CNY": raise ValueError("Expected CNY pricing")
        spec = config.get("models", {}).get(model)
        if not spec:
            return {**result, "reason": "model_not_priced"}
        incoming, outgoing, cached = [usage.get(k) for k in ("input_tokens", "output_tokens", "cache_read_tokens")]
        if any(type(v) is not int or v < 0 for v in (incoming, outgoing, cached)):
            return {**result, "reason": "missing_or_invalid_usage"}
        if cached > incoming or usage.get("cache_creation_tokens", 0) not in (None, 0) or config.get("cache_mode") != "implicit":
            return {**result, "reason": "unsupported_cache_usage"}
        tier = next((t for t in spec["tiers"] if incoming <= t[0]), None)
        if tier is None: return {**result, "reason": "outside_price_tiers"}
        _, input_rate, cache_rate, output_rate = tier
        d = lambda value: Decimal(str(value))
        if any(not d(r).is_finite() or d(r) < 0 for r in (input_rate,cache_rate,output_rate)): raise ValueError("Invalid price")
        prompt = (d(incoming-cached)*d(input_rate) + d(cached)*d(cache_rate))/d(1000000)
        completion = d(outgoing)*d(output_rate)/d(1000000)
        fx = d(config["cny_per_usd"])
        if not fx.is_finite() or fx <= 0: raise ValueError("invalid FX")
        result.update(status="estimated", profile=config["profile"], source=spec["source"],
                      verified_on=spec.get("verified_on", config["verified_on"]), input_tier_max=tier[0],
                      rates_cny_per_million={"input":input_rate,"cache_read":cache_rate,"output":output_rate},
                      prompt_cny=float(prompt), completion_cny=float(completion), total_cny=float(prompt+completion),
                      prompt_usd=float(prompt/fx), completion_usd=float(completion/fx), total_usd=float((prompt+completion)/fx),
                      cny_per_usd=float(fx), fx_date=config["fx_date"], fx_source=config["fx_source"],
                      note="北京区公开价估算；不含赠送额度、折扣或套餐；不是实际扣款。汇率为注明日期的展示换算。")
        return result
    except Exception as error:
        return {**result, "reason": "pricing_config_error", "error_type": type(error).__name__}

def cost_attributes(cost):
    attrs = {"cost.status":cost["status"], "cost.details_json":json.dumps(cost,ensure_ascii=False)}
    if cost["status"] == "estimated":
        attrs.update({"cost.currency":"CNY", "cost.total_cny":cost["total_cny"],
                      "llm.cost.prompt":cost["prompt_usd"],"llm.cost.completion":cost["completion_usd"],
                      "llm.cost.total":cost["total_usd"]})
    return attrs

def summarize_cost(costs):
    known = [c for c in costs if c.get("status") == "estimated"]
    return {"status":"estimated" if costs and len(known)==len(costs) else "partial" if known else "unknown",
            "currency":"CNY", "known_total_cny":sum(c["total_cny"] for c in known) if known else None,
            "missing_requests":len(costs)-len(known), "priced_requests":len(known)}
