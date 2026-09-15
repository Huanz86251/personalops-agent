from __future__ import annotations

import json
import logging
import os
import requests

from urllib.parse import (
    urlsplit,
)
from contextlib import (
    ExitStack,
    contextmanager,
)
from dataclasses import (
    asdict,
    is_dataclass,
)
from threading import (
    Lock,
)
from typing import (
    Any,
    Iterator,
    Mapping,
)


logger = logging.getLogger(
    "agent"
)


DEFAULT_PHOENIX_PROJECT_NAME = (
    "Agent Tasks"
)

DEFAULT_PHOENIX_UI_URL = (
    "http://127.0.0.1:6007"
)

DEFAULT_PHOENIX_COLLECTOR_ENDPOINT = (
    "http://127.0.0.1:6007/v1/traces"
)
DEFAULT_PHOENIX_PROTOCOL = (
    "http/protobuf"
)

DEFAULT_PHOENIX_TRACE_PROFILE = (
    "curated"
)


_SETUP_LOCK = Lock()

_TRACER_PROVIDER: Any | None = None

_TRACER: Any | None = None

_SETUP_ATTEMPTED = False

def _is_loopback_endpoint(
    endpoint: str,
) -> bool:
    """判断Phoenix Collector是否运行在本机。"""

    try:
        hostname = (
            urlsplit(
                endpoint
            )
            .hostname
        )

    except ValueError:
        return False

    if not isinstance(
        hostname,
        str,
    ):
        return False

    return (
        hostname
        .strip()
        .casefold()
        in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
    )
def _read_bool_environment(
    name: str,
    default: bool,
) -> bool:
    """读取布尔类型环境变量。"""

    raw_value = os.getenv(
        name
    )

    if raw_value is None:
        return default

    normalized_value = (
        raw_value
        .strip()
        .lower()
    )

    return normalized_value in {
        "1",
        "true",
        "yes",
        "on",
    }


def _make_json_safe(
    value: Any,
) -> Any:
    """把常见Python对象转换成可以写入Trace的结构。"""

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(
        model_dump
    ):
        try:
            return _make_json_safe(
                model_dump()
            )

        except Exception:
            pass

    if is_dataclass(
        value
    ):
        try:
            return _make_json_safe(
                asdict(
                    value
                )
            )

        except Exception:
            pass

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): (
                _make_json_safe(
                    item
                )
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            _make_json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    return str(
        value
    )


def _to_otel_attribute_value(
    value: Any,
) -> (
    str
    | int
    | float
    | bool
):
    """把自定义属性转换成OpenTelemetry允许的简单类型。

    复杂结构会保存成JSON字符串。

    真正需要JSON查看器展示的主要数据，
    后续会通过span.set_input和span.set_output保存。
    """

    safe_value = (
        _make_json_safe(
            value
        )
    )

    if isinstance(
        safe_value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return safe_value

    if safe_value is None:
        return "null"

    return json.dumps(
        safe_value,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )


def _normalize_http_endpoint(
    endpoint: str,
    protocol: str,
) -> str:
    """为HTTP OTLP端点补全/v1/traces。"""

    normalized_endpoint = (
        endpoint
        .strip()
        .rstrip("/")
    )

    if (
        protocol
        == "http/protobuf"

        and not normalized_endpoint.endswith(
            "/v1/traces"
        )
    ):
        normalized_endpoint += (
            "/v1/traces"
        )

    return normalized_endpoint


def get_phoenix_project_name() -> str:
    """读取Phoenix项目名称。"""

    return (
        os.getenv(
            "PHOENIX_PROJECT",
            DEFAULT_PHOENIX_PROJECT_NAME,
        )
        .strip()
        or DEFAULT_PHOENIX_PROJECT_NAME
    )


def get_phoenix_ui_url() -> str:
    """读取Phoenix网页地址。"""

    return (
        os.getenv(
            "PHOENIX_UI_URL",
            DEFAULT_PHOENIX_UI_URL,
        )
        .strip()
        .rstrip("/")
        or DEFAULT_PHOENIX_UI_URL
    )


def get_phoenix_collector_endpoint() -> str:
    """读取Phoenix Trace接收地址。"""

    protocol = (
        os.getenv(
            "PHOENIX_OTEL_PROTOCOL",
            DEFAULT_PHOENIX_PROTOCOL,
        )
        .strip()
        or DEFAULT_PHOENIX_PROTOCOL
    )

    endpoint = (
        os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            DEFAULT_PHOENIX_COLLECTOR_ENDPOINT,
        )
        .strip()
        or DEFAULT_PHOENIX_COLLECTOR_ENDPOINT
    )

    return _normalize_http_endpoint(
        endpoint=endpoint,
        protocol=protocol,
    )


def get_phoenix_trace_profile() -> str:
    """Return the selected Phoenix trace detail profile.

    curated is the default. The legacy full value remains accepted, but both
    use the same application-owned model/tool leaves and semantic boundaries.
    Framework/SDK auto-instrumentation is not enabled in either profile.
    """

    profile = (
        os.getenv(
            "PHOENIX_TRACE_PROFILE",
            DEFAULT_PHOENIX_TRACE_PROFILE,
        )
        .strip()
        .lower()
    )

    if profile not in {
        "full",
        "curated",
    }:
        logger.warning(
                "Invalid PHOENIX_TRACE_PROFILE: %s; using curated.",
            profile,
        )
        return DEFAULT_PHOENIX_TRACE_PROFILE

    return profile


def _curated_span_name(name: str) -> str:
    """Stable role/stage names shared by every supported trace profile."""

    aliases = {
        "hard_supervisor": "Scheduler / Plan Decision",
        "main_agent.run": "General Agent",
        "hard_replanner": "Scheduler / Replan Decision",
        "hard_final_reviewer": "Scheduler / Final Decision",
        "hard_code_scheduler": "Scheduler / Code Recovery Decision",
        "hard_worker_leader": "Scheduler / Worker Guidance",
        "conversation_turn": "Conversation / Turn",
        "conversation_title_generation": "Conversation / Title",
        "memory_retrieval": "Memory / Recall",
        "memory_write_buffer": "Memory / Buffer User Message",
        "memory.write_gate": "Memory / Write Eligibility",
        "memory.write_gate.memoperator": "Memory / Local Write Classifier",
        "memory.read_gate": "Memory / Read Eligibility",
        "memory.dense_retrieval": "Memory / Dense Search",
        "memory.bm25_retrieval": "Memory / Lexical Search",
        "memory.graph_expansion": "Memory / Graph Expansion",
        "memory.seed_ranking": "Memory / Rank Seeds",
        "memory.final_ranking": "Memory / Rank Results",
        "memory.duplicate_check": "Memory / Check Duplicates",
        "memory.store_new": "Memory / Save Record",
        "memory.apply_resolution": "Memory / Apply Resolution",
        "memory.retire_old": "Memory / Retire Record",
        "memory.cloud_resolution": "Memory / Resolve Conflict",
        "memory.relation_gate": "Memory / Check Relation",
        "memory.resolution_retrieval": "Memory / Retrieve Conflict Evidence",
        "toolset_routing": "Tool Router / Rank Capabilities",
        "toolset_selection": "Tool Router / Select Capabilities",
        "local_router": "Local Model / Tool Router",
        "local_embedding.encode": "Local Model / Embedding",
        "local_cross_encoder.predict": "Local Model / Reranker Scores",
        "execution_budget.model_limit": "Budget / Model Limit",
        "execution_budget.tool_filter": "Budget / Filter Tools",
        "message_history.repair": "Context / Repair Message History",
        "conversation_summary.update": "Conversation Summary / Update",
    }
    if name in aliases:
        return aliases[name]
    prefix = "step_report.step_"
    if name.startswith(prefix):
        return "Step Reporter / Step " + name[len(prefix):]
    if name.startswith("model_call.round_"):
        return "Agent / Model Round " + name.removeprefix("model_call.round_")
    if name.startswith("memory."):
        return "Memory / " + name.removeprefix("memory.").replace("_", " ").replace(".", " / ").title()
    return name


def setup_observability() -> Any | None:
    """注册Phoenix和LangChain/LangGraph自动追踪。

    这个函数可以安全地被重复调用，
    实际注册过程只会执行一次。

    Phoenix没有安装或初始化失败时，
    Agent仍然可以继续运行。
    """

    global _SETUP_ATTEMPTED
    global _TRACER_PROVIDER
    global _TRACER

    if not _read_bool_environment(
        "PHOENIX_TRACING_ENABLED",
        default=True,
    ):
        return None

    if _TRACER_PROVIDER is not None:
        return _TRACER_PROVIDER

    if _SETUP_ATTEMPTED:
        return None

    with _SETUP_LOCK:
        if _TRACER_PROVIDER is not None:
            return _TRACER_PROVIDER

        if _SETUP_ATTEMPTED:
            return None

        _SETUP_ATTEMPTED = True

        try:
            from phoenix.otel import (
                HTTPSpanExporter,
                register,
            )

            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
            )

        except ImportError:
            logger.warning(
                "Phoenix观测组件尚未安装，"
                "本次运行不会记录Phoenix Trace。"
            )

            return None

        protocol = (
            os.getenv(
                "PHOENIX_OTEL_PROTOCOL",
                DEFAULT_PHOENIX_PROTOCOL,
            )
            .strip()
            or DEFAULT_PHOENIX_PROTOCOL
        )

        if protocol not in {
            "http/protobuf",
            "grpc",
        }:
            logger.warning(
                "PHOENIX_OTEL_PROTOCOL配置无效：%s；"
                "本次使用http/protobuf。",
                protocol,
            )

            protocol = (
                DEFAULT_PHOENIX_PROTOCOL
            )

        collector_endpoint = (
            get_phoenix_collector_endpoint()
        )

        project_name = (
            get_phoenix_project_name()
        )

        trace_profile = (
            get_phoenix_trace_profile()
        )

        # One application-owned callback records model/tool leaves. Framework
        # plus SDK auto-instrumentation would duplicate the same request.
        auto_instrument = False

        try:
            use_direct_local_exporter = (
                    protocol
                    == "http/protobuf"

                    and _is_loopback_endpoint(
                collector_endpoint
            )
            )

            if use_direct_local_exporter:
                # register仍然负责：
                #
                # 1. 创建Phoenix TracerProvider；
                # 2. 设置project_name；
                # 3. 注册为全局TracerProvider；
                # 4. 自动启用LangChain/LangGraph追踪。
                #
                # verbose=False是因为下面会替换
                # register自动创建的默认Exporter，
                # 不希望先打印一份过时配置。
                tracer_provider = register(
                    project_name=(
                        project_name
                    ),

                    endpoint=(
                        collector_endpoint
                    ),

                    protocol=(
                        protocol
                    ),

                    batch=False,

                    auto_instrument=auto_instrument,

                    verbose=False,
                )

                # 为Phoenix单独创建Requests Session。
                #
                # trust_env=False只影响这个Session：
                #
                # - 不读取HTTP_PROXY；
                # - 不读取HTTPS_PROXY；
                # - 不读取ALL_PROXY；
                # - 不读取NO_PROXY；
                #
                # 因此Phoenix会直接访问本机6006端口，
                # 但不会改变Telegram、DeepSeek和Web Search
                # 的网络代理行为。
                phoenix_session = (
                    requests.Session()
                )

                phoenix_session.trust_env = (
                    False
                )

                phoenix_exporter = (
                    HTTPSpanExporter(
                        endpoint=(
                            collector_endpoint
                        ),

                        session=(
                            phoenix_session
                        ),
                    )
                )

                phoenix_processor = (
                    BatchSpanProcessor(
                        span_exporter=(
                            phoenix_exporter
                        )
                    )
                )

                # Phoenix TracerProvider会自动移除并关闭
                # register创建的默认SpanProcessor，
                # 然后使用我们这个只针对本机直连的Processor。
                tracer_provider.add_span_processor(
                    phoenix_processor
                )

                network_mode = (
                    "direct_loopback"
                )

            else:
                # 远程Phoenix保持正常代理行为。
                tracer_provider = register(
                    project_name=(
                        project_name
                    ),

                    endpoint=(
                        collector_endpoint
                    ),

                    protocol=(
                        protocol
                    ),

                    batch=True,

                    auto_instrument=auto_instrument,

                    verbose=False,
                )

                network_mode = (
                    "environment_default"
                )

            tracer = (
                tracer_provider
                .get_tracer(
                    "agentnew"
                )
            )

        except Exception:
            logger.exception(
                "Phoenix观测初始化失败，"
                "Agent将继续以无Trace模式运行。"
            )

            return None

        _TRACER_PROVIDER = (
            tracer_provider
        )

        _TRACER = tracer

        logger.info(
            "Phoenix观测已启用 | "
            "project=%s | ui=%s | "
            "network=%s | processor=batch | profile=%s",

            project_name,

            get_phoenix_ui_url(),

            network_mode,

            trace_profile,
        )

        return _TRACER_PROVIDER


def get_tracer() -> Any | None:
    """返回当前Phoenix Tracer。"""

    if _TRACER is None:
        setup_observability()

    return _TRACER


def is_observability_enabled() -> bool:
    """判断Phoenix Tracer是否已经成功初始化。"""

    return get_tracer() is not None


@contextmanager
def trace_context(
    *,
    session_id: str | None = None,
    metadata: Mapping[
        str,
        Any,
    ] | None = None,
    tags: list[str] | None = None,
) -> Iterator[None]:
    """为一组Span附加Session、Metadata和Tags。

    后续ConversationRuntime会使用thread_id作为session_id，
    让同一个Conversation的多轮Trace在Phoenix中归组。
    """

    if get_tracer() is None:
        yield
        return

    try:
        from phoenix.otel import (
            using_metadata,
            using_session,
            using_tags,
        )

    except ImportError:
        yield
        return

    with ExitStack() as stack:
        if (
            isinstance(
                session_id,
                str,
            )
            and session_id.strip()
        ):
            stack.enter_context(
                using_session(
                    session_id=(
                        session_id.strip()
                    )
                )
            )

        if metadata:
            stack.enter_context(
                using_metadata(
                    _make_json_safe(
                        metadata
                    )
                )
            )

        if tags:
            normalized_tags = [
                tag.strip()
                for tag in tags
                if (
                    isinstance(
                        tag,
                        str,
                    )
                    and tag.strip()
                )
            ]

            if normalized_tags:
                stack.enter_context(
                    using_tags(
                        normalized_tags
                    )
                )

        yield


@contextmanager
def trace_span(
    name: str,
    *,
    kind: str = "chain",
    input_value: Any = None,
    attributes: Mapping[
        str,
        Any,
    ] | None = None,
) -> Iterator[Any | None]:
    """创建一个带异常状态处理的结构化Phoenix Span。

    Phoenix或OpenTelemetry不可用时，
    自动降级为无Trace模式。

    业务代码抛出的异常必须继续向上传递，
    不能被观测层误吞掉。
    """

    tracer = get_tracer()

    if tracer is None:
        yield None
        return

    # ImportError只允许覆盖真正的模块导入。
    #
    # 不能把下面的业务代码也放在这个try里面，
    # 否则业务中的ImportError会被误认为
    # OpenTelemetry没有安装。
    try:
        from opentelemetry.trace import (
            Status,
            StatusCode,
        )

    except ImportError:
        yield None
        return

    with tracer.start_as_current_span(
        _curated_span_name(name),

        openinference_span_kind=(
            kind
        ),
    ) as span:

        if input_value is not None:
            span.set_input(
                _make_json_safe(
                    input_value
                )
            )

        if attributes:
            for (
                attribute_name,
                attribute_value,
            ) in attributes.items():

                span.set_attribute(
                    str(
                        attribute_name
                    ),

                    _to_otel_attribute_value(
                        attribute_value
                    ),
                )

        try:
            from usage_accounting import usage_scope
            from trace_presentation import timing_scope
            with timing_scope(span), usage_scope(span):
                yield span

        except BaseException as error:
            # 记录业务异常，
            # 但绝对不能吞掉或改变异常类型。
            span.record_exception(
                error
            )

            span.set_status(
                Status(
                    StatusCode.ERROR,

                    str(
                        error
                    ),
                )
            )

            raise

        else:
            span.set_status(
                Status(
                    StatusCode.OK
                )
            )

def set_span_input(
    span: Any | None,
    value: Any,
) -> None:
    """设置Span的结构化输入。"""

    if span is None:
        return

    span.set_input(
        _make_json_safe(
            value
        )
    )


def set_span_output(
    span: Any | None,
    value: Any,
) -> None:
    """设置Span的结构化输出。"""

    if span is None:
        return

    span.set_output(
        _make_json_safe(
            value
        )
    )

    # RAG reading view is a separate attribute; canonical output is untouched.
    if isinstance(value, dict) and (
        (isinstance(value.get("injected_context"), str) and value["injected_context"].startswith("[文档检索资料"))
        or (isinstance(value.get("results"), list) and any(
            isinstance(row, dict) and "source" in row and "text" in row
            for row in value["results"]))
    ):
        from trace_chat import readable_content
        span.set_attribute("llm.output_messages.0.message.role", "assistant")
        span.set_attribute("llm.output_messages.0.message.content", readable_content(value))
        span.set_attribute("rag.presentation_only", True)


def set_span_attributes(
    span: Any | None,
    **attributes: Any,
) -> None:
    """为Span补充少量可筛选属性。"""

    if span is None:
        return

    for (
        attribute_name,
        attribute_value,
    ) in attributes.items():

        span.set_attribute(
            attribute_name,

            _to_otel_attribute_value(
                attribute_value
            ),
        )


def shutdown_observability() -> None:
    """刷新并关闭Phoenix TracerProvider。"""

    global _TRACER_PROVIDER
    global _TRACER

    tracer_provider = (
        _TRACER_PROVIDER
    )

    _TRACER_PROVIDER = None
    _TRACER = None

    if tracer_provider is None:
        return

    shutdown = getattr(
        tracer_provider,
        "shutdown",
        None,
    )

    if not callable(
        shutdown
    ):
        return

    try:
        shutdown()

    except Exception:
        logger.exception(
            "关闭Phoenix观测组件时发生异常"
        )
