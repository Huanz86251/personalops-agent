import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv
from model_roles import RoleModelSettings, load_role_models

from delivery.models import DeliveryApprovalMode
from feishu_exports import ExportPolicy


SUPPORTED_PROVIDERS = {
    "deepseek",
    "openai",
    "anthropic",
    "compatible",
    "qwen",
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
    "scheduler_summary_trigger_tokens": ("SCHEDULER_SUMMARY_TRIGGER_TOKENS", 20000, 1000, 200000),
    # 单次Supervisor或Replanner最多生成多少个Step。
    "max_steps_per_plan": (
        "PLANNING_MAX_STEPS_PER_PLAN",
        8,
        1,
        8,
    ),

    "max_total_steps": (
        "PLANNING_MAX_TOTAL_STEPS",
        16,
        1,
        16,
    ),

    # 当前架构允许0至2次Replan，仍共享模型总预算。
    "max_replans": (
        "PLANNING_MAX_REPLANS",
        2,
        0,
        2,
    ),

    # 单个Step整体模型预算。
    "max_step_model_rounds": (
        "PLANNING_MAX_STEP_MODEL_ROUNDS",
        21,
        2,
        30,
    ),

    # 多次attempt共同分享的Executor模型预算。
    "max_step_executor_rounds": (
        "PLANNING_MAX_STEP_EXECUTOR_ROUNDS",
        16,
        1,
        28,
    ),

    # Step Reporter预算。
    "max_step_report_rounds": (
        "PLANNING_MAX_STEP_REPORT_ROUNDS",
        5,
        1,
        5,
    ),

    # 多次attempt共同分享的工具预算。
    "max_step_tool_calls": (
        "PLANNING_MAX_STEP_TOOL_CALLS",
        30,
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
        60,
        4,
        120,
    ),

    "max_plan_tool_calls": (
        "PLANNING_MAX_PLAN_TOOL_CALLS",
        90,
        1,
        240,
    ),

    # Final Reviewer优先把事实缺口交回最后一个Worker；这部分有独立边界，
    # 不因正常Plan预算刚好耗尽而被静默跳过。
    "max_final_worker_repair_rounds": (
        "PLANNING_MAX_FINAL_WORKER_REPAIR_ROUNDS",
        3,
        0,
        3,
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


# 这些值是架构硬上限，不允许通过.env继续放大。
#
# 运行时配置可以把实际并发调低，
# 但不能越过这里的资源与一致性边界。
WEB_SEARCH_HARD_MAX_PARALLELISM = 3
PLAYWRIGHT_HARD_MAX_SESSIONS = 3
CODE_HARD_MAX_WRITERS = 1
CODE_REVIEW_DEFAULT_REPAIR_ROUNDS = 2
CODE_REVIEW_HARD_MAX_REPAIR_ROUNDS = 3
WORKER_PROGRESS_MIN_TOOL_CALLS = 2
WORKER_PROGRESS_DEFAULT_TOOL_CALLS = 4
WORKER_PROGRESS_HARD_MAX_TOOL_CALLS = 8
WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS = 4
WORKER_FINALIZATION_HARD_MAX_MODEL_ROUNDS = 4
WORKER_SCHEMA_REPAIR_DEFAULT_ROUNDS = 3
WORKER_SCHEMA_REPAIR_HARD_MAX_ROUNDS = 3
from file_limits import TASK_FILE_MAX_MIB

WEB_DOWNLOAD_DEFAULT_MAX_FILE_MIB = TASK_FILE_MAX_MIB
WEB_DOWNLOAD_HARD_MAX_FILE_MIB = TASK_FILE_MAX_MIB
WORKER_WORKSPACE_DEFAULT_RETENTION_MINUTES = 10 * 24 * 60
WORKER_WORKSPACE_MAX_RETENTION_MINUTES = 30 * 24 * 60


RUNTIME_CONCURRENCY_INTEGER_OPTIONS: dict[
    str,
    tuple[
        str,
        int,
        int,
        int,
    ],
] = {
    "web_search_max_parallelism": (
        "RUNTIME_WEB_SEARCH_MAX_PARALLELISM",
        3,
        1,
        WEB_SEARCH_HARD_MAX_PARALLELISM,
    ),

    "playwright_max_sessions": (
        "RUNTIME_PLAYWRIGHT_MAX_SESSIONS",
        3,
        1,
        PLAYWRIGHT_HARD_MAX_SESSIONS,
    ),

    # 当前Code Agent只能有一个可写工作区拥有者。
    # 以后可以增加只读Reviewer，但不能绕过这个writer上限。
    "code_max_writers": (
        "RUNTIME_CODE_MAX_WRITERS",
        1,
        1,
        CODE_HARD_MAX_WRITERS,
    ),
}


WORKER_RUNTIME_INTEGER_OPTIONS: dict[
    str,
    tuple[
        str,
        int,
        int,
        int,
    ],
] = {
    "code_review_max_repair_rounds": (
        "RUNTIME_CODE_REVIEW_MAX_REPAIR_ROUNDS",
        CODE_REVIEW_DEFAULT_REPAIR_ROUNDS,
        1,
        CODE_REVIEW_HARD_MAX_REPAIR_ROUNDS,
    ),

    "progress_every_tool_calls": (
        "RUNTIME_WORKER_PROGRESS_EVERY_TOOL_CALLS",
        WORKER_PROGRESS_DEFAULT_TOOL_CALLS,
        WORKER_PROGRESS_MIN_TOOL_CALLS,
        WORKER_PROGRESS_HARD_MAX_TOOL_CALLS,
    ),

    "finalization_model_rounds": (
        "RUNTIME_WORKER_FINALIZATION_MODEL_ROUNDS",
        WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS,
        1,
        WORKER_FINALIZATION_HARD_MAX_MODEL_ROUNDS,
    ),

    # 提交工具的结构化参数被Harness拒绝后，仍由原角色修表。
    # 这部分不占普通执行轮次，也不触发独立Reviewer。
    "schema_repair_max_rounds": (
        "RUNTIME_WORKER_SCHEMA_REPAIR_MAX_ROUNDS",
        WORKER_SCHEMA_REPAIR_DEFAULT_ROUNDS,
        1,
        WORKER_SCHEMA_REPAIR_HARD_MAX_ROUNDS,
    ),

    "web_download_max_file_mib": (
        "RUNTIME_WEB_DOWNLOAD_MAX_FILE_MIB",
        WEB_DOWNLOAD_DEFAULT_MAX_FILE_MIB,
        1,
        WEB_DOWNLOAD_HARD_MAX_FILE_MIB,
    ),

    "leadership_single_worker_reports": (
        "RUNTIME_LEADERSHIP_SINGLE_WORKER_REPORTS",
        2,
        1,
        10,
    ),

    "leadership_multi_worker_reports": (
        "RUNTIME_LEADERSHIP_MULTI_WORKER_REPORTS",
        1,
        1,
        5,
    ),

    "leadership_silence_timeout_seconds": (
        "RUNTIME_LEADERSHIP_SILENCE_TIMEOUT_SECONDS",
        60,
        10,
        600,
    ),

    "workspace_retention_minutes": (
        "RUNTIME_WORKER_WORKSPACE_RETENTION_MINUTES",
        WORKER_WORKSPACE_DEFAULT_RETENTION_MINUTES,
        10,
        WORKER_WORKSPACE_MAX_RETENTION_MINUTES,
    ),
}


CODE_SANDBOX_INTEGER_OPTIONS: dict[
    str,
    tuple[str, int, int, int],
] = {
    "memory_mb": (
        "RUNTIME_CODE_SANDBOX_MEMORY_MB",
        1536,
        256,
        4096,
    ),
    "cpu_count": (
        "RUNTIME_CODE_SANDBOX_CPU_COUNT",
        2,
        1,
        4,
    ),
    "pids_limit": (
        "RUNTIME_CODE_SANDBOX_PIDS_LIMIT",
        128,
        32,
        256,
    ),
    "execute_timeout_seconds": (
        "RUNTIME_CODE_SANDBOX_EXECUTE_TIMEOUT_SECONDS",
        120,
        10,
        600,
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

    scheduler_summary_trigger_tokens: int = 20000
    max_final_worker_repair_rounds: int = 3


@dataclass(
    frozen=True,
)
class RuntimeConcurrencySettings:
    """保存跨Worker共享的运行时并发策略。"""

    web_search_max_parallelism: int
    playwright_max_sessions: int
    code_max_writers: int


@dataclass(
    frozen=True,
)
class WorkerRuntimeSettings:
    """保存Web与Code Worker共享的运行时控制配置。"""

    code_review_max_repair_rounds: int
    progress_every_tool_calls: int
    finalization_model_rounds: int
    schema_repair_max_rounds: int
    web_download_max_file_mib: int
    leadership_single_worker_reports: int
    leadership_multi_worker_reports: int
    leadership_silence_timeout_seconds: int
    workspace_retention_minutes: int


@dataclass(
    frozen=True,
)
class CodeSandboxSettings:
    """保存Code Worker/Reviewer共享的Docker资源边界。"""

    image: str
    wsl_distribution: str
    auto_build: bool
    memory_mb: int
    cpu_count: int
    pids_limit: int
    execute_timeout_seconds: int


@dataclass(
    frozen=True,
)
class DeliverySettings:
    """Control whether reviewed files require a human promotion decision."""

    approval_mode: DeliveryApprovalMode


@dataclass(
    frozen=True,
)
class PromptInjectionGuardSettings:
    """Local two-stage filtering for text returned by tools."""

    enabled: bool
    primary_model: str
    primary_onnx_file: str
    primary_threshold: float
    primary_window_tokens: int
    primary_overlap_tokens: int
    primary_batch_size: int
    secondary_model: str
    secondary_window_tokens: int
    secondary_overlap_tokens: int
    secondary_batch_size: int
    secondary_device: str


@dataclass(
    frozen=True,
)
class EmailMCPSettings:
    """本地只读邮件MCP的连接配置。"""

    enabled: bool
    address: str
    auth_code: str = field(repr=False)
    imap_host: str
    imap_port: int
    folder: str
    drafts_folder: str
    secure: bool

    @property
    def is_configured(self) -> bool:
        """账号与授权码齐全时才允许真正连接。"""

        return bool(
            self.address
            and self.auth_code
            and self.imap_host
            and self.folder
            and self.drafts_folder
        )


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
    # Executor、Reporter、Supervisor和Final Reviewer共用该上限；
    # 批量记忆提取在下方使用独立的宽松上限。
    cloud_llm_max_tokens: int

    # 渐进式记忆提取使用低推理开销，第二轮可能批量输出较多记录，
    # 因此使用独立且更宽的极端保护上限。
    memory_extraction_max_tokens: int

    memory_embedding_model: str
    memory_reranker_model: str
    # CrossEncoder经过Sigmoid后的最低相关性分数。
    memory_reranker_threshold: float
    # 工具组路由只采用绝对最低分；低于它时继续第二轮或模型回退。
    toolset_routing_threshold: float
    memory_bm25_limit: int
    memory_model_device: str
    memory_model_cache_dir: Path

    # 按批次临时加载的长期记忆写入门控。
    memory_write_gate_enabled: bool
    memory_write_gate_model: str
    memory_write_gate_device: str
    memory_write_gate_batch_size: int
    memory_write_gate_max_length: int
    memory_write_gate_threshold: float

    # 通过门控后的云端渐进式两轮抽取。默认关闭，避免部署升级后
    # 在未确认费用前自动恢复云调用；启用后按批次处理持久化候选。
    memory_extraction_enabled: bool
    memory_extraction_batch_size: int

    # 本地轻量记忆路由模型。
    memory_router_enabled: bool
    memory_router_model_repo: str
    memory_router_model_filename: str
    memory_router_context_length: int
    memory_router_max_tokens: int

    # Planning Graph集中配置。
    planning: PlanningSettings

    # Web、浏览器和代码Worker的统一并发边界。
    runtime_concurrency: RuntimeConcurrencySettings

    # Web与Code Worker共享的进度上报策略。
    worker_runtime: WorkerRuntimeSettings

    # Code Worker和Reviewer的Docker沙盒策略。
    code_sandbox: CodeSandboxSettings

    # Reviewer通过后的最终Conversation Workspace交付策略。
    delivery: DeliverySettings

    # Tool结果在进入Worker模型前使用的本地两级注入检测。
    prompt_injection_guard: PromptInjectionGuardSettings

    # 可选的本地只读IMAP MCP。关闭时不会启动Node子进程，
    # 也不会把邮箱工具注册给模型。
    email_mcp: EmailMCPSettings

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

    # Scheduler 与普通执行/报告模型分别控制；动态技能选择沿用其角色配置。
    scheduler_thinking_enabled: bool = True
    llm_thinking_enabled: bool = False
    summary_llm_provider: str = "qwen"
    summary_llm_model: str = "qwen3.7-flash"
    extraction_llm_provider: str = "qwen"
    extraction_llm_model: str = "qwen3.7-flash"

    # Empty identities/roots deliberately disable file export until configured.
    feishu_export: ExportPolicy = field(default_factory=ExportPolicy)
    role_models: dict[str, RoleModelSettings] = field(default_factory=dict)

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


def _read_bool_environment(
    name: str,
    *,
    default: bool,
) -> bool:
    raw_value = os.getenv(
        name,
        "true" if default else "false",
    ).strip().lower()
    if raw_value not in {"true", "false"}:
        raise RuntimeError(f"{name}只支持true或false。")
    return raw_value == "true"


def _read_float_environment(
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """读取并验证浮点环境变量。"""

    raw_value = os.getenv(
        name,
        str(default),
    ).strip()

    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{name}必须是数字。"
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


def _load_runtime_concurrency_settings(
) -> RuntimeConcurrencySettings:
    """读取并检查跨Worker的并发配置。"""

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
        in RUNTIME_CONCURRENCY_INTEGER_OPTIONS.items()
    }

    return RuntimeConcurrencySettings(
        **values
    )


def _load_worker_runtime_settings(
) -> WorkerRuntimeSettings:
    """读取并检查通用Worker运行时控制配置。"""

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
        in WORKER_RUNTIME_INTEGER_OPTIONS.items()
    }

    return WorkerRuntimeSettings(
        **values
    )


def _load_code_sandbox_settings(
) -> CodeSandboxSettings:
    """读取CODE Docker镜像、自动构建与资源上限。"""

    values = {
        field_name: _read_int_environment(
            environment_name,
            default=default,
            minimum=minimum,
            maximum=maximum,
        )
        for field_name, (
            environment_name,
            default,
            minimum,
            maximum,
        ) in CODE_SANDBOX_INTEGER_OPTIONS.items()
    }
    image = os.getenv(
        "RUNTIME_CODE_SANDBOX_IMAGE",
        "personalops-code-sandbox:py312-v2",
    ).strip()
    wsl_distribution = os.getenv(
        "RUNTIME_CODE_SANDBOX_WSL_DISTRIBUTION",
        "Ubuntu",
    ).strip()
    if not image:
        raise RuntimeError("RUNTIME_CODE_SANDBOX_IMAGE不能为空。")
    if not wsl_distribution:
        raise RuntimeError(
            "RUNTIME_CODE_SANDBOX_WSL_DISTRIBUTION不能为空。"
        )
    return CodeSandboxSettings(
        image=image,
        wsl_distribution=wsl_distribution,
        auto_build=_read_bool_environment(
            "RUNTIME_CODE_SANDBOX_AUTO_BUILD",
            default=True,
        ),
        **values,
    )


def _load_delivery_settings() -> DeliverySettings:
    raw_mode = os.getenv(
        "RUNTIME_DELIVERY_APPROVAL_MODE",
        DeliveryApprovalMode.AUTO.value,
    ).strip().lower()
    try:
        approval_mode = DeliveryApprovalMode(raw_mode)
    except ValueError as error:
        raise RuntimeError(
            "RUNTIME_DELIVERY_APPROVAL_MODE只支持human或auto。"
        ) from error
    return DeliverySettings(approval_mode=approval_mode)


def _load_prompt_injection_guard_settings() -> PromptInjectionGuardSettings:
    """Load the documented Wolf/Qwen guard protocol with bounded overrides."""

    return PromptInjectionGuardSettings(
        enabled=_read_bool_environment(
            "PROMPT_INJECTION_GUARD_ENABLED",
            default=True,
        ),
        primary_model=os.getenv(
            "PROMPT_INJECTION_PRIMARY_MODEL",
            "patronus-studio/wolf-defender-prompt-injection-small",
        ).strip(),
        primary_onnx_file=os.getenv(
            "PROMPT_INJECTION_PRIMARY_ONNX_FILE",
            "onnx/int8_int4_embeddings/model.onnx",
        ).strip(),
        # Wolf Defender Small v2 uses 0.5 for its published document results.
        primary_threshold=_read_float_environment(
            "PROMPT_INJECTION_PRIMARY_THRESHOLD",
            default=0.5,
            minimum=0.0,
            maximum=1.0,
        ),
        primary_window_tokens=_read_int_environment(
            "PROMPT_INJECTION_PRIMARY_WINDOW_TOKENS",
            default=2048,
            minimum=128,
            maximum=2048,
        ),
        # The model card's long-document protocol uses a 64-token overlap.
        primary_overlap_tokens=_read_int_environment(
            "PROMPT_INJECTION_PRIMARY_OVERLAP_TOKENS",
            default=64,
            minimum=0,
            maximum=512,
        ),
        primary_batch_size=_read_int_environment(
            "PROMPT_INJECTION_PRIMARY_BATCH_SIZE",
            default=8,
            minimum=1,
            maximum=64,
        ),
        secondary_model=os.getenv(
            "PROMPT_INJECTION_SECONDARY_MODEL",
            "Qwen/Qwen3Guard-Gen-0.6B",
        ).strip(),
        secondary_window_tokens=_read_int_environment(
            "PROMPT_INJECTION_SECONDARY_WINDOW_TOKENS",
            default=512,
            minimum=128,
            maximum=2048,
        ),
        secondary_overlap_tokens=_read_int_environment(
            "PROMPT_INJECTION_SECONDARY_OVERLAP_TOKENS",
            default=40,
            minimum=0,
            maximum=256,
        ),
        secondary_batch_size=_read_int_environment(
            "PROMPT_INJECTION_SECONDARY_BATCH_SIZE",
            default=4,
            minimum=1,
            maximum=16,
        ),
        secondary_device=os.getenv(
            "PROMPT_INJECTION_SECONDARY_DEVICE",
            "auto",
        ).strip().lower(),
    )


def load_settings() -> Settings:
    """读取通用.env与本机邮箱配置，并检查当前模型配置。"""

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
    # 邮箱授权码与项目通用配置分开保存。该文件被.gitignore中的
    # `.env.*` 覆盖，不会进入版本库；已存在的进程环境仍拥有最高优先级。
    load_dotenv(
        dotenv_path=(
            env_path.parent
            / ".env.email.local"
        ),
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

    memory_write_gate_enabled_text = os.getenv(
        "MEMORY_WRITE_GATE_ENABLED",
        "true",
    ).strip().lower()

    if memory_write_gate_enabled_text not in {"true", "false"}:
        raise RuntimeError(
            "MEMORY_WRITE_GATE_ENABLED只支持：true或false。"
        )

    memory_write_gate_enabled = (
        memory_write_gate_enabled_text == "true"
    )

    memory_extraction_enabled_text = os.getenv(
        "MEMORY_EXTRACTION_ENABLED",
        "true",
    ).strip().lower()
    if memory_extraction_enabled_text not in {"true", "false"}:
        raise RuntimeError(
            "MEMORY_EXTRACTION_ENABLED只支持：true或false。"
        )
    memory_extraction_enabled = memory_extraction_enabled_text == "true"

    try:
        memory_write_gate_batch_size = int(
            os.getenv(
                "MEMORY_WRITE_GATE_BATCH_SIZE",
                "3",
            ).strip()
        )

        memory_write_gate_max_length = int(
            os.getenv(
                "MEMORY_WRITE_GATE_MAX_LENGTH",
                "256",
            ).strip()
        )

        memory_extraction_batch_size = int(
            os.getenv(
                "MEMORY_EXTRACTION_BATCH_SIZE",
                "10",
            ).strip()
        )

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
            "记忆本地模型的批次、上下文和输出配置必须是整数。"
        ) from error

    if not 1 <= memory_write_gate_batch_size <= 32:
        raise RuntimeError(
            "MEMORY_WRITE_GATE_BATCH_SIZE必须在1到32之间。"
        )

    if not 32 <= memory_write_gate_max_length <= 2048:
        raise RuntimeError(
            "MEMORY_WRITE_GATE_MAX_LENGTH必须在32到2048之间。"
        )

    try:
        memory_write_gate_threshold = float(os.getenv(
            "MEMORY_WRITE_GATE_THRESHOLD",
            "0.690976",
        ).strip())
    except ValueError as error:
        raise RuntimeError(
            "MEMORY_WRITE_GATE_THRESHOLD必须是0到1之间的数字。"
        ) from error
    if not 0.0 <= memory_write_gate_threshold <= 1.0:
        raise RuntimeError(
            "MEMORY_WRITE_GATE_THRESHOLD必须在0到1之间。"
        )

    if not 1 <= memory_extraction_batch_size <= 32:
        raise RuntimeError(
            "MEMORY_EXTRACTION_BATCH_SIZE必须在1到32之间。"
        )

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
            "qwen",
        ).strip().lower(),

        llm_model=os.getenv(
            "LLM_MODEL",
            "qwen3.7-flash",
        ).strip(),

        hard_llm_provider=os.getenv(
            "HARD_LLM_PROVIDER",
            "qwen",
        ).strip().lower(),

        hard_llm_model=os.getenv(
            "HARD_LLM_MODEL",
            "qwen3.8-flash",
        ).strip(),
        summary_llm_provider=os.getenv("SUMMARY_LLM_PROVIDER", "qwen").strip().lower(),
        summary_llm_model=os.getenv("SUMMARY_LLM_MODEL", "qwen3.7-flash").strip(),
        extraction_llm_provider=os.getenv("EXTRACTION_LLM_PROVIDER", "qwen").strip().lower(),
        extraction_llm_model=os.getenv("EXTRACTION_LLM_MODEL", "qwen3.7-flash").strip(),
        scheduler_thinking_enabled=_read_bool_environment("SCHEDULER_THINKING_ENABLED", default=True),
        llm_thinking_enabled=_read_bool_environment("LLM_THINKING_ENABLED", default=False),
        cloud_llm_max_tokens=(
            _read_int_environment(
                "CLOUD_LLM_MAX_TOKENS",

                default=5000,

                minimum=512,

                maximum=16000,
            )
        ),
        memory_extraction_max_tokens=(
            _read_int_environment(
                "MEMORY_EXTRACTION_MAX_TOKENS",

                default=16000,

                minimum=2048,

                maximum=65536,
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
                "maidalun1020/"
                "bce-reranker-base_v1"
            ),
        ).strip(),

        memory_reranker_threshold=(
            _read_float_environment(
                "MEMORY_RERANKER_THRESHOLD",

                # BCE官方模型卡建议使用0.35或0.4
                # 过滤低质量候选；默认采用更保守的0.4。
                default=0.4,

                minimum=0.0,

                maximum=1.0,
            )
        ),

        toolset_routing_threshold=(
            _read_float_environment(
                "TOOLSET_ROUTING_THRESHOLD",
                default=0.4,
                minimum=0.0,
                maximum=1.0,
            )
        ),
        memory_bm25_limit=(
            _read_int_environment(
                "MEMORY_BM25_LIMIT",
                default=8,
                minimum=1,
                maximum=100,
            )
        ),

        memory_model_device=os.getenv(
            "MEMORY_MODEL_DEVICE",
            "auto",
        ).strip().lower(),

        memory_model_cache_dir=(
            model_cache_dir
        ),

        memory_write_gate_enabled=(
            memory_write_gate_enabled
        ),

        memory_write_gate_model=os.getenv(
            "MEMORY_WRITE_GATE_MODEL",
            "chris0809/memoperator-0.6b-memory-write-gate",
        ).strip(),

        memory_write_gate_device=os.getenv(
            "MEMORY_WRITE_GATE_DEVICE",
            "cpu",
        ).strip().lower(),

        memory_write_gate_batch_size=(
            memory_write_gate_batch_size
        ),

        memory_write_gate_max_length=(
            memory_write_gate_max_length
        ),

        memory_write_gate_threshold=(
            memory_write_gate_threshold
        ),

        memory_extraction_enabled=memory_extraction_enabled,

        memory_extraction_batch_size=memory_extraction_batch_size,

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

        runtime_concurrency=(
            _load_runtime_concurrency_settings()
        ),

        worker_runtime=(
            _load_worker_runtime_settings()
        ),

        code_sandbox=(
            _load_code_sandbox_settings()
        ),

        delivery=(
            _load_delivery_settings()
        ),
        prompt_injection_guard=(
            _load_prompt_injection_guard_settings()
        ),
        email_mcp=EmailMCPSettings(
            enabled=_read_bool_environment(
                "EMAIL_MCP_ENABLED",
                default=False,
            ),
            address=os.getenv(
                "EMAIL_MCP_ADDRESS",
                "",
            ).strip(),
            auth_code=os.getenv(
                "EMAIL_MCP_AUTH_CODE",
                "",
            ).strip(),
            imap_host=os.getenv(
                "EMAIL_MCP_IMAP_HOST",
                "imap.qq.com",
            ).strip(),
            imap_port=_read_int_environment(
                "EMAIL_MCP_IMAP_PORT",
                default=993,
                minimum=1,
                maximum=65535,
            ),
            folder=os.getenv(
                "EMAIL_MCP_FOLDER",
                "INBOX",
            ).strip(),
            drafts_folder=os.getenv(
                "EMAIL_MCP_DRAFTS_FOLDER",
                "Drafts",
            ).strip(),
            secure=_read_bool_environment(
                "EMAIL_MCP_SECURE",
                default=True,
            ),
        ),
        feishu_export=ExportPolicy.from_env(),

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

    if settings.memory_write_gate_enabled:
        if not settings.memory_write_gate_model:
            raise RuntimeError(
                "启用Write Gate时，MEMORY_WRITE_GATE_MODEL不能为空。"
            )

        if settings.memory_write_gate_device not in {"cpu", "cuda"}:
            raise RuntimeError(
                "MEMORY_WRITE_GATE_DEVICE只支持：cpu或cuda。"
            )

    if settings.email_mcp.enabled:
        missing_server_names = [
            name
            for name, value in (
                ("EMAIL_MCP_IMAP_HOST", settings.email_mcp.imap_host),
                ("EMAIL_MCP_FOLDER", settings.email_mcp.folder),
                ("EMAIL_MCP_DRAFTS_FOLDER", settings.email_mcp.drafts_folder),
            )
            if not value
        ]
        if missing_server_names:
            raise RuntimeError(
                "启用本地邮箱MCP时缺少服务器配置："
                + ", ".join(missing_server_names)
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
    settings = replace(settings, role_models=load_role_models(settings))

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
