import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


SUPPORTED_PROVIDERS = {
    "deepseek",
    "openai",
    "anthropic",
    "compatible",
}
PLANNING_INTEGER_OPTIONS: dict[
    str,
    tuple[
        str,
        int,
        int,
        int,
    ],
] = {
    # 单次Supervisor或Replanner最多生成多少个Step。
    "max_steps_per_plan": (
        "PLANNING_MAX_STEPS_PER_PLAN",
        2,
        1,
        8,
    ),

    "max_total_steps": (
        "PLANNING_MAX_TOTAL_STEPS",
        2,
        1,
        16,
    ),

    # 当前架构允许0或1次Replan。
    "max_replans": (
        "PLANNING_MAX_REPLANS",
        1,
        0,
        1,
    ),

    # 单个Step整体模型预算。
    "max_step_model_rounds": (
        "PLANNING_MAX_STEP_MODEL_ROUNDS",
        10,
        2,
        30,
    ),

    # 多次attempt共同分享的Executor模型预算。
    "max_step_executor_rounds": (
        "PLANNING_MAX_STEP_EXECUTOR_ROUNDS",
        8,
        1,
        28,
    ),

    # Step Reporter预算。
    "max_step_report_rounds": (
        "PLANNING_MAX_STEP_REPORT_ROUNDS",
        2,
        1,
        4,
    ),

    # 多次attempt共同分享的工具预算。
    "max_step_tool_calls": (
        "PLANNING_MAX_STEP_TOOL_CALLS",
        15,
        1,
        60,
    ),

    # 单个Step最多尝试几次。
    "max_step_attempts": (
        "PLANNING_MAX_STEP_ATTEMPTS",
        2,
        1,
        4,
    ),

    # 整个用户请求的模型与工具总预算。
    "max_plan_model_rounds": (
        "PLANNING_MAX_PLAN_MODEL_ROUNDS",
        30,
        4,
        120,
    ),

    "max_plan_tool_calls": (
        "PLANNING_MAX_PLAN_TOOL_CALLS",
        45,
        1,
        240,
    ),

    # Hard节点保留的最近原始对话。
    "hard_recent_dialogue_turns": (
        "PLANNING_HARD_RECENT_DIALOGUE_TURNS",
        5,
        1,
        20,
    ),

    "hard_recent_dialogue_max_chars": (
        "PLANNING_HARD_RECENT_DIALOGUE_MAX_CHARS",
        6000,
        1000,
        30000,
    ),

    # 较早Conversation的Rolling Summary配置。
    "conversation_summary_trigger_turns": (
        "PLANNING_CONVERSATION_SUMMARY_TRIGGER_TURNS",
        10,
        2,
        50,
    ),

    "conversation_summary_max_chars": (
        "PLANNING_CONVERSATION_SUMMARY_MAX_CHARS",
        6000,
        1000,
        30000,
    ),

    # Simple Executor独立Step Thread的摘要配置。
    "executor_summary_trigger_tokens": (
        "PLANNING_EXECUTOR_SUMMARY_TRIGGER_TOKENS",
        5000,
        1000,
        30000,
    ),

    "executor_summary_trigger_messages": (
        "PLANNING_EXECUTOR_SUMMARY_TRIGGER_MESSAGES",
        18,
        6,
        100,
    ),

    "executor_summary_keep_messages": (
        "PLANNING_EXECUTOR_SUMMARY_KEEP_MESSAGES",
        10,
        4,
        40,
    ),
}


@dataclass(
    frozen=True,
)
class PlanningSettings:
    """保存Planning Graph预算和上下文配置。"""

    max_steps_per_plan: int
    max_total_steps: int
    max_replans: int

    max_step_model_rounds: int
    max_step_executor_rounds: int
    max_step_report_rounds: int
    max_step_tool_calls: int
    max_step_attempts: int

    max_plan_model_rounds: int
    max_plan_tool_calls: int

    hard_recent_dialogue_turns: int
    hard_recent_dialogue_max_chars: int

    conversation_summary_trigger_turns: int
    conversation_summary_max_chars: int

    executor_summary_trigger_tokens: int
    executor_summary_trigger_messages: int
    executor_summary_keep_messages: int

@dataclass(
    frozen=True,
)
class Settings:
    """保存程序启动时需要的配置。"""

    env_path: Path

    # 飞书国内版企业自建应用凭证。
    feishu_app_id: str

    feishu_app_secret: str = field(
        repr=False,
    )

    llm_provider: str
    llm_model: str

    hard_llm_provider: str
    hard_llm_model: str

    # 所有云端模型单次调用的最大输出Token。
    #
    # 当前演示项目不为Executor、Reporter、
    # Supervisor和Final Reviewer拆分独立上限。
    cloud_llm_max_tokens: int

    memory_embedding_model: str
    memory_reranker_model: str
    memory_model_device: str
    memory_model_cache_dir: Path

    # 本地轻量记忆路由模型。
    memory_router_enabled: bool
    memory_router_model_repo: str
    memory_router_model_filename: str
    memory_router_context_length: int
    memory_router_max_tokens: int

    # Planning Graph集中配置。
    planning: PlanningSettings

    deepseek_api_key: str = field(
        default="",
        repr=False,
    )

    openai_api_key: str = field(
        default="",
        repr=False,
    )

    anthropic_api_key: str = field(
        default="",
        repr=False,
    )

    compatible_api_key: str = field(
        default="",
        repr=False,
    )

    compatible_base_url: str = ""

def _read_int_environment(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """读取并验证整数环境变量。"""

    raw_value = os.getenv(
        name,
        str(default),
    ).strip()

    try:
        value = int(
            raw_value
        )

    except ValueError as error:
        raise RuntimeError(
            f"{name}必须是整数。"
        ) from error

    if not minimum <= value <= maximum:
        raise RuntimeError(
            f"{name}必须在"
            f"{minimum}到{maximum}之间。"
        )

    return value


def _load_planning_settings(
) -> PlanningSettings:
    """读取并检查Planning配置。"""

    values = {
        field_name: (
            _read_int_environment(
                environment_name,

                default=default,

                minimum=minimum,

                maximum=maximum,
            )
        )

        for (
            field_name,
            (
                environment_name,
                default,
                minimum,
                maximum,
            ),
        )
        in PLANNING_INTEGER_OPTIONS.items()
    }

    planning = PlanningSettings(
        **values
    )

    if (
        planning.max_total_steps
        < planning.max_steps_per_plan
    ):
        raise RuntimeError(
            "PLANNING_MAX_TOTAL_STEPS"
            "不能小于"
            "PLANNING_MAX_STEPS_PER_PLAN。"
        )

    allocated_step_rounds = (
        planning.max_step_executor_rounds
        + planning.max_step_report_rounds
    )

    if (
        allocated_step_rounds
        > planning.max_step_model_rounds
    ):
        raise RuntimeError(
            "PLANNING_MAX_STEP_EXECUTOR_ROUNDS"
            "与"
            "PLANNING_MAX_STEP_REPORT_ROUNDS"
            "之和不能超过"
            "PLANNING_MAX_STEP_MODEL_ROUNDS。"
        )

    if (
        planning.max_plan_model_rounds
        < planning.max_step_model_rounds
    ):
        raise RuntimeError(
            "PLANNING_MAX_PLAN_MODEL_ROUNDS"
            "不能小于"
            "PLANNING_MAX_STEP_MODEL_ROUNDS。"
        )

    if (
        planning.executor_summary_keep_messages
        >= planning.executor_summary_trigger_messages
    ):
        raise RuntimeError(
            "PLANNING_EXECUTOR_SUMMARY_KEEP_MESSAGES"
            "必须小于"
            "PLANNING_EXECUTOR_SUMMARY_TRIGGER_MESSAGES。"
        )

    return planning
def load_settings() -> Settings:
    """读取 .env，并检查当前选择的模型配置。"""

    env_path = (
        Path(__file__)
        .resolve()
        .parent
        / ".env"
    )

    load_dotenv(
        dotenv_path=env_path,
        override=False,
    )
    model_cache_value = os.getenv(
        "MEMORY_MODEL_CACHE_DIR",
        ".models",
    ).strip()

    model_cache_dir = (
        Path(model_cache_value)
        .expanduser()
    )

    if not model_cache_dir.is_absolute():
        model_cache_dir = (
            env_path.parent
            / model_cache_dir
        )

    model_cache_dir = (
        model_cache_dir
        .resolve()
    )
    memory_router_enabled_text = os.getenv(
        "MEMORY_ROUTER_ENABLED",
        "true",
    ).strip().lower()

    if memory_router_enabled_text not in {
        "true",
        "false",
    }:
        raise RuntimeError(
            "MEMORY_ROUTER_ENABLED只支持："
            "true或false。"
        )

    memory_router_enabled = (
            memory_router_enabled_text
            == "true"
    )

    try:
        memory_router_context_length = int(
            os.getenv(
                "MEMORY_ROUTER_CONTEXT_LENGTH",
                "512",
            ).strip()
        )

        memory_router_max_tokens = int(
            os.getenv(
                "MEMORY_ROUTER_MAX_TOKENS",
                "16",
            ).strip()
        )

    except ValueError as error:
        raise RuntimeError(
            "MEMORY_ROUTER_CONTEXT_LENGTH和"
            "MEMORY_ROUTER_MAX_TOKENS"
            "必须是整数。"
        ) from error

    if memory_router_context_length < 256:
        raise RuntimeError(
            "MEMORY_ROUTER_CONTEXT_LENGTH"
            "不能小于256。"
        )

    if not 1 <= memory_router_max_tokens <= 128:
        raise RuntimeError(
            "MEMORY_ROUTER_MAX_TOKENS"
            "必须在1到128之间。"
        )
    settings = Settings(
        env_path=env_path,

        feishu_app_id=os.getenv(
            "FEISHU_APP_ID",
            "",
        ).strip(),

        feishu_app_secret=os.getenv(
            "FEISHU_APP_SECRET",
            "",
        ).strip(),

        llm_provider=os.getenv(
            "LLM_PROVIDER",
            "deepseek",
        ).strip().lower(),

        llm_model=os.getenv(
            "LLM_MODEL",
            "",
        ).strip(),

        hard_llm_provider=os.getenv(
            "HARD_LLM_PROVIDER",
            "deepseek",
        ).strip().lower(),

        hard_llm_model=os.getenv(
            "HARD_LLM_MODEL",
            "",
        ).strip(),
        cloud_llm_max_tokens=(
            _read_int_environment(
                "CLOUD_LLM_MAX_TOKENS",

                default=5000,

                minimum=512,

                maximum=16000,
            )
        ),
        memory_embedding_model=os.getenv(
            "MEMORY_EMBEDDING_MODEL",
            (
                "Alibaba-NLP/"
                "gte-multilingual-base"
            ),
        ).strip(),

        memory_reranker_model=os.getenv(
            "MEMORY_RERANKER_MODEL",
            (
                "Alibaba-NLP/"
                "gte-multilingual-reranker-base"
            ),
        ).strip(),

        memory_model_device=os.getenv(
            "MEMORY_MODEL_DEVICE",
            "auto",
        ).strip().lower(),

        memory_model_cache_dir=(
            model_cache_dir
        ),

        memory_router_enabled=(
            memory_router_enabled
        ),

        memory_router_model_repo=os.getenv(
            "MEMORY_ROUTER_MODEL_REPO",
            "Qwen/Qwen3-0.6B-GGUF",
        ).strip(),

        memory_router_model_filename=os.getenv(
            "MEMORY_ROUTER_MODEL_FILENAME",
            "Qwen3-0.6B-Q8_0.gguf",
        ).strip(),

        memory_router_context_length=(
            memory_router_context_length
        ),

        memory_router_max_tokens=(
            memory_router_max_tokens
        ),
        planning=(
            _load_planning_settings()
        ),

        deepseek_api_key=os.getenv(
            "DEEPSEEK_API_KEY",
            "",
        ).strip(),

        openai_api_key=os.getenv(
            "OPENAI_API_KEY",
            "",
        ).strip(),

        anthropic_api_key=os.getenv(
            "ANTHROPIC_API_KEY",
            "",
        ).strip(),

        compatible_api_key=os.getenv(
            "COMPATIBLE_API_KEY",
            "",
        ).strip(),

        compatible_base_url=os.getenv(
            "COMPATIBLE_BASE_URL",
            "",
        ).strip(),
    )

    provider_api_keys = {
        "deepseek": settings.deepseek_api_key,
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "compatible": settings.compatible_api_key,
    }

    provider_key_names = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "compatible": "COMPATIBLE_API_KEY",
    }
    if settings.memory_router_enabled:
        if not settings.memory_router_model_repo:
            raise RuntimeError(
                "启用本地记忆路由器时，"
                "MEMORY_ROUTER_MODEL_REPO不能为空。"
            )

        if not settings.memory_router_model_filename:
            raise RuntimeError(
                "启用本地记忆路由器时，"
                "MEMORY_ROUTER_MODEL_FILENAME不能为空。"
            )

        if not (
                settings
                        .memory_router_model_filename
                        .lower()
                        .endswith(
                    ".gguf"
                )
        ):
            raise RuntimeError(
                "MEMORY_ROUTER_MODEL_FILENAME"
                "必须是.gguf文件。"
            )
    missing_names = []

    if not settings.feishu_app_id:
        missing_names.append(
            "FEISHU_APP_ID"
        )

    if not settings.feishu_app_secret:
        missing_names.append(
            "FEISHU_APP_SECRET"
        )
    model_configs = [
        (
            "LLM_PROVIDER",
            "LLM_MODEL",

            settings.llm_provider,
            settings.llm_model,
        ),

        (
            "HARD_LLM_PROVIDER",
            "HARD_LLM_MODEL",

            settings.hard_llm_provider,
            settings.hard_llm_model,
        ),
    ]

    for (
        provider_variable,
        model_variable,
        provider,
        model,
    ) in model_configs:

        if provider not in SUPPORTED_PROVIDERS:
            raise RuntimeError(
                f"{provider_variable} 使用了"
                f"不支持的供应商：{provider!r}"
            )

        if not model:
            missing_names.append(
                model_variable
            )

        if not provider_api_keys[provider]:
            missing_names.append(
                provider_key_names[provider]
            )

        if (
            provider == "compatible"
            and not settings.compatible_base_url
        ):
            missing_names.append(
                "COMPATIBLE_BASE_URL"
            )

    if missing_names:
        missing_names = sorted(
            set(missing_names)
        )

        missing_text = ", ".join(
            missing_names
        )

        raise RuntimeError(
            "缺少必要的环境变量："
            f"{missing_text}。"
            f"请检查文件：{env_path}"
        )

    return settings