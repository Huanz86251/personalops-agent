"""Independent cloud model settings; no clients or network calls at load time."""
from dataclasses import dataclass, field
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROLE_DEFAULTS = {
    "skill_selector": "regular",
    "general": "regular", "web": "regular", "code": "hard",
    "code_reviewer": "hard", "reporter": "regular", "web_reporter": "regular",
    "scope_resolver": "hard", "scheduler": "hard", "code_scheduler": "hard", "replanner": "hard",
    "final_reviewer": "hard", "worker_leader": "hard", "title": "regular",
    "summary": "summary", "general_summary": "summary", "web_summary": "summary",
    "code_summary": "summary", "code_reviewer_summary": "summary",
    "extraction": "extraction",
}
PROVIDER_KEY_ENVS = {
    "openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY", "qwen": "DASHSCOPE_API_KEY",
    "compatible": "COMPATIBLE_API_KEY",
}
PROVIDER_URL_ENVS = {
    "openai": "OPENAI_BASE_URL", "deepseek": "DEEPSEEK_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL", "qwen": "DASHSCOPE_BASE_URL",
    "compatible": "COMPATIBLE_BASE_URL",
}


@dataclass(frozen=True)
class RoleModelSettings:
    provider: str
    model: str
    api_key_env: str
    api_key: str = field(repr=False)
    base_url: str = ""
    thinking_enabled: bool = False
    reasoning_effort: str | None = None
    max_tokens: int = 5000
    timeout_seconds: int = 120
    max_retries: int = 2
    token_limit_parameter: str = "max_completion_tokens"
    extra_body: dict[str, Any] = field(default_factory=dict, repr=False)


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip() or str(default))
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def load_role_models(settings) -> dict[str, RoleModelSettings]:
    """Resolve each role independently; only summary roles inherit summary defaults."""
    thinking_config = json.loads((Path(__file__).parent / 'config/model_thinking.json').read_text(encoding='utf-8'))
    flash_budget = thinking_config['qwen3.7-flash']['thinking_budget']
    if type(flash_budget) is not int or not 0 <= flash_budget <= 262144:
        raise ValueError('qwen3.7-flash thinking_budget must be an integer from 0 to 262144')
    selector_budget = thinking_config['qwen3.7-flash'].get('skill_selector_thinking_budget', 0)
    if type(selector_budget) is not int or not 0 <= selector_budget <= 262144:
        raise ValueError('skill_selector_thinking_budget must be an integer from 0 to 262144')
    defaults = {
        "regular": (settings.llm_provider, settings.llm_model, settings.llm_thinking_enabled),
        "hard": (settings.hard_llm_provider, settings.hard_llm_model, settings.scheduler_thinking_enabled),
        "summary": (settings.summary_llm_provider, settings.summary_llm_model, False),
        "extraction": (settings.extraction_llm_provider, settings.extraction_llm_model, False),
    }
    roles = {}
    for role, group in ROLE_DEFAULTS.items():
        prefix = role.upper() + "_LLM_"
        default_provider, default_model, default_thinking = defaults[group]
        provider = (os.getenv(prefix + "PROVIDER", "").strip() or default_provider).lower()
        if provider not in PROVIDER_KEY_ENVS:
            raise ValueError(f"{prefix}PROVIDER must be one of {', '.join(PROVIDER_KEY_ENVS)}")
        model = os.getenv(prefix + "MODEL", "").strip()
        if not model and provider != default_provider:
            raise ValueError(f"{prefix}MODEL is required when changing provider")
        model = model or default_model
        if not model:
            raise ValueError(f"{prefix}MODEL is required")
        key_env = os.getenv(prefix + "API_KEY_ENV", "").strip() or PROVIDER_KEY_ENVS[provider]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
            raise ValueError(f"{prefix}API_KEY_ENV must be an environment variable name, not a key")
        key = os.getenv(key_env, "").strip()
        active = role != "extraction" or settings.memory_extraction_enabled
        if active and not key:
            raise ValueError(f"{role}: missing API key environment variable {key_env}")
        base_url = (os.getenv(prefix + "BASE_URL", "").strip()
                    or os.getenv(PROVIDER_URL_ENVS[provider], "").strip())
        if active and provider in {"qwen", "compatible"} and not base_url:
            raise ValueError(f"{prefix}BASE_URL is required for {provider}")
        if base_url:
            url = urlsplit(base_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError(f"{prefix}BASE_URL must be an HTTP(S) API base URL without credentials or query")
        thinking = os.getenv(prefix + "THINKING_ENABLED", str(False if role == "skill_selector" else default_thinking)).strip().lower()
        if thinking not in {"true", "false"}:
            raise ValueError(f"{prefix}THINKING_ENABLED must be true or false")
        default_effort = (
            "medium" if role == "scope_resolver" and provider == "qwen" and model.startswith("qwen3.8-")
            else "low" if provider == "qwen" and model.startswith("qwen3.8-") and thinking == "true"
            else ""
        )
        effort = os.getenv(prefix + "REASONING_EFFORT", default_effort).strip() or None
        if effort not in {None, "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"{prefix}REASONING_EFFORT is invalid")
        token_param = os.getenv(prefix + "TOKEN_LIMIT_PARAMETER", "").strip() or (
            "max_tokens" if provider == "compatible" or (provider == "qwen" and model.startswith("qwen-flash"))
            else "max_completion_tokens"
        )
        if token_param not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"{prefix}TOKEN_LIMIT_PARAMETER is invalid")
        try:
            extra = json.loads(os.getenv(prefix + "EXTRA_BODY_JSON", "{}") or "{}")
        except json.JSONDecodeError:
            raise ValueError(f"{prefix}EXTRA_BODY_JSON must be a JSON object") from None
        if not isinstance(extra, dict) or set(extra) & {"model", "messages", "stream", "max_tokens", "max_completion_tokens"}:
            raise ValueError(f"{prefix}EXTRA_BODY_JSON cannot override model/messages/stream/token limits")
        is_flash37 = provider == 'qwen' and (model == 'qwen3.7-flash' or model.startswith('qwen3.7-flash-'))
        max_output = _integer(
            prefix + "MAX_TOKENS",
            settings.memory_extraction_max_tokens if role == "extraction"
            else 512 if role == "skill_selector"
            else 4096 if role == "scope_resolver"
            else settings.cloud_llm_max_tokens,
            1, 131072,
        )
        if is_flash37:
            role_budget = selector_budget if role == 'skill_selector' and flash_budget else flash_budget
            # Single authoritative switch for all Flash 3.7 roles, including selectors.
            thinking = 'true' if role_budget else 'false'
            effort = None
            for option in ('enable_thinking', 'thinking_budget', 'reasoning_effort'):
                extra.pop(option, None)
            extra['enable_thinking'] = bool(role_budget)
            if role_budget:
                extra['thinking_budget'] = role_budget
                # Preserve the selector's 512-token answer allowance.
                if not os.getenv(prefix + 'MAX_TOKENS', '').strip() and role == 'skill_selector':
                    max_output += role_budget
                if max_output <= role_budget:
                    raise ValueError(f'{prefix}MAX_TOKENS must exceed thinking_budget to leave room for the answer')
            token_param = 'max_completion_tokens'
        roles[role] = RoleModelSettings(
            provider=provider, model=model, api_key_env=key_env, api_key=key,
            base_url=base_url, thinking_enabled=thinking == "true", reasoning_effort=effort,
            max_tokens=max_output,
            timeout_seconds=_integer(prefix + "TIMEOUT_SECONDS", 180 if group == "hard" else 120, 1, 3600),
            max_retries=_integer(prefix + "MAX_RETRIES", 2, 0, 5),
            token_limit_parameter=token_param, extra_body=extra,
        )
    return roles
