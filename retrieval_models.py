import asyncio
import gc
import logging
import re
from collections.abc import (
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch
from langchain_core.embeddings import (
    Embeddings,
)
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)
from huggingface_hub import (
    hf_hub_download,
)
from llama_cpp import (
    Llama,
    LlamaRAMCache,
)
from observability import (
    set_span_attributes,
    set_span_output,
    trace_span,
)
logger = logging.getLogger(
    "agent"
)


GTE_EMBEDDING_DIMENSIONS = 768
LOCAL_MODEL_TRACE_PREVIEW_MAX_CHARS = 240

@dataclass(frozen=True)
class RerankResult:
    """表示Cross-Encoder重排后的一条结果。"""

    index: int
    text: str
    score: float

def _compact_local_model_trace_text(
    text: str,
) -> str:
    """压缩本地模型Span中的输入文本预览。

    完整业务文本由外层Memory或Tool Selection Span保存。

    底层物理模型Span只保留短预览，
    避免同一内容在Phoenix中重复占用大量空间。
    """

    normalized_text = (
        " ".join(
            text
            .strip()
            .split()
        )
    )

    if (
        len(
            normalized_text
        )
        <= LOCAL_MODEL_TRACE_PREVIEW_MAX_CHARS
    ):
        return normalized_text

    head_length = (
        LOCAL_MODEL_TRACE_PREVIEW_MAX_CHARS
        // 2
    )

    tail_length = (
        LOCAL_MODEL_TRACE_PREVIEW_MAX_CHARS
        - head_length
    )

    return (
        normalized_text[
            :head_length
        ]
        + "\n...\n"
        + normalized_text[
            -tail_length:
        ]
    )

def _resolve_device(
    requested_device: str,
) -> str:
    """根据配置和本机能力选择推理设备。"""

    normalized_device = (
        requested_device
        .strip()
        .lower()
    )

    if normalized_device == "auto":
        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    if normalized_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MEMORY_MODEL_DEVICE配置为cuda，"
                "但当前PyTorch无法使用CUDA。"
            )

        return "cuda"

    if normalized_device == "cpu":
        return "cpu"

    raise ValueError(
        "MEMORY_MODEL_DEVICE只支持："
        "auto、cuda、cpu。"
    )
class LangChainEmbeddingAdapter(
    Embeddings
):
    """把本地Embedding模型适配成LangChain接口。"""

    def __init__(
        self,
        manager: "RetrievalModelManager",
    ) -> None:
        self.manager = manager

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self.manager.embed_documents(
            texts
        )

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self.manager.embed_query(
            text
        )

    async def aembed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return await (
            self.manager
            .aembed_documents(
                texts
            )
        )

    async def aembed_query(
        self,
        text: str,
    ) -> list[float]:
        return await (
            self.manager
            .aembed_query(
                text
            )
        )

class RetrievalModelManager:
    """管理本地Embedding和Reranker模型。"""

    def __init__(
            self,
            embedding_model_name: str,
            reranker_model_name: str,
            cache_dir: str | Path,
            device: str = "auto",
            embedding_batch_size: int = 16,
            reranker_batch_size: int = 8,
            embedding_max_length: int = 1024,
            reranker_max_length: int = 1024,

            router_enabled: bool = False,
            router_model_repo: str = "",
            router_model_filename: str = "",
            router_context_length: int = 512,
            router_max_tokens: int = 16,
    ) -> None:
        self.embedding_model_name = (
            embedding_model_name
        )

        self.reranker_model_name = (
            reranker_model_name
        )

        self.cache_dir = (
            Path(cache_dir)
            .expanduser()
            .resolve()
        )

        self.device = _resolve_device(
            device
        )

        self.embedding_batch_size = (
            embedding_batch_size
        )

        self.reranker_batch_size = (
            reranker_batch_size
        )

        self.embedding_max_length = (
            embedding_max_length
        )

        self.reranker_max_length = (
            reranker_max_length
        )
        self.router_enabled = (
            router_enabled
        )

        self.router_model_repo = (
            router_model_repo
        )

        self.router_model_filename = (
            router_model_filename
        )

        self.router_context_length = (
            router_context_length
        )

        self.router_max_tokens = (
            router_max_tokens
        )
        self._embedding_model: (
            SentenceTransformer
            | None
        ) = None

        self._reranker_model: (
            CrossEncoder
            | None
        ) = None
        self._router_model: (
                Llama
                | None
        ) = None
        self._embedding_lock = Lock()
        self._reranker_lock = Lock()

        # llama.cpp中的同一个模型实例
        # 不应被多个线程同时调用。
        self._router_lock = Lock()

        self.langchain_embeddings = (
            LangChainEmbeddingAdapter(
                self
            )
        )

    def load(
            self,
    ) -> None:
        """加载并常驻本地检索模型和轻量路由模型。"""

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch_dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        def _load_local_first(
                model_label: str,
                loader,
        ):
            """优先从本地缓存加载，缺失时才联网下载。"""

            try:
                loaded_object = loader(
                    True
                )

            except Exception as local_error:
                logger.info(
                    "%s本地缓存不可用，"
                    "将尝试从Hugging Face下载 | "
                    "error=%s: %s",

                    model_label,

                    type(
                        local_error
                    ).__name__,

                    local_error,
                )

            else:
                logger.info(
                    "%s已从本地缓存加载",
                    model_label,
                )

                return (
                    loaded_object,
                    "local_cache",
                )

            try:
                loaded_object = loader(
                    False
                )

            except Exception as online_error:
                raise RuntimeError(
                    f"{model_label}本地缓存不可用，"
                    "并且从Hugging Face下载失败。"
                ) from online_error

            logger.info(
                "%s已从Hugging Face下载并加载",
                model_label,
            )

            return (
                loaded_object,
                "huggingface_download",
            )
        if self._embedding_model is None:
            logger.info(
                "开始加载本地Embedding模型 | "
                "model=%s | device=%s",
                self.embedding_model_name,
                self.device,
            )

            (
                embedding_model,
                embedding_load_source,
            ) = _load_local_first(
                "Embedding模型",

                lambda local_files_only: (
                    SentenceTransformer(
                        self.embedding_model_name,

                        device=self.device,

                        cache_folder=str(
                            self.cache_dir
                        ),

                        trust_remote_code=True,

                        local_files_only=(
                            local_files_only
                        ),

                        model_kwargs={
                            "torch_dtype": (
                                torch_dtype
                            ),
                        },
                    )
                ),
            )

            embedding_model.max_seq_length = (
                self.embedding_max_length
            )

            self._embedding_model = (
                embedding_model
            )

            logger.info(
                "本地Embedding模型加载完成 | "
                "model=%s",
                self.embedding_model_name,
            )

        if self._reranker_model is None:
            logger.info(
                "开始加载本地Reranker模型 | "
                "model=%s | device=%s",
                self.reranker_model_name,
                self.device,
            )

            (
                reranker_model,
                reranker_load_source,
            ) = _load_local_first(
                "Reranker模型",

                lambda local_files_only: (
                    CrossEncoder(
                        self.reranker_model_name,

                        device=self.device,

                        cache_folder=str(
                            self.cache_dir
                        ),

                        trust_remote_code=True,

                        local_files_only=(
                            local_files_only
                        ),

                        max_length=(
                            self.reranker_max_length
                        ),

                        model_kwargs={
                            "torch_dtype": (
                                torch_dtype
                            ),
                        },
                    )
                ),
            )

            self._reranker_model = (
                reranker_model
            )

            logger.info(
                "本地Reranker模型加载完成 | "
                "model=%s",
                self.reranker_model_name,
            )

        if (
                self.router_enabled
                and self._router_model is None
        ):
            logger.info(
                "开始加载本地Memory Router | "
                "repo=%s | filename=%s | "
                "device=gpu",

                self.router_model_repo,
                self.router_model_filename,
            )

            try:
                router_cache_dir = (
                        self.cache_dir
                        / "memory_router"
                )

                router_cache_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                (
                    router_model_path,
                    router_load_source,
                ) = _load_local_first(
                    "Memory Router模型文件",

                    lambda local_files_only: (
                        hf_hub_download(
                            repo_id=(
                                self.router_model_repo
                            ),

                            filename=(
                                self.router_model_filename
                            ),

                            cache_dir=str(
                                router_cache_dir
                            ),

                            local_files_only=(
                                local_files_only
                            ),
                        )
                    ),
                )

                router_model = Llama(
                    model_path=(
                        router_model_path
                    ),

                    # 固定使用CPU，
                    # 不与Embedding和Reranker争抢显存。
                    n_gpu_layers=-1,

                    n_ctx=(
                        self.router_context_length
                    ),

                    n_batch=min(
                        self.router_context_length,
                        512,
                    ),

                    # 固定随机种子，
                    # 让分类结果尽可能稳定。
                    seed=42,

                    use_mmap=True,

                    verbose=False,
                )
                router_model.set_cache(
                    LlamaRAMCache(
                        capacity_bytes=(
                                512
                                * 1024
                                * 1024
                        ),
                    )
                )
            except Exception:
                logger.exception(
                    "本地Memory Router加载失败，"
                    "程序将继续运行；"
                    "后续记忆判断会保守升级到云端"
                )

            else:
                self._router_model = (
                    router_model
                )

                logger.info(
                    "本地Memory Router加载完成 | "
                    "repo=%s | filename=%s | "
                    "context_length=%s",

                    self.router_model_repo,
                    self.router_model_filename,
                    self.router_context_length,
                )

        logger.info(
            "本地模型加载阶段完成 | "
            "embedding_dims=%s | "
            "router_enabled=%s | "
            "router_loaded=%s",

            GTE_EMBEDDING_DIMENSIONS,
            self.router_enabled,
            self._router_model is not None,
        )

    async def aload(
        self,
    ) -> None:
        """在线程中加载模型，避免阻塞事件循环。"""

        await asyncio.to_thread(
            self.load
        )

    @staticmethod
    def _extract_router_labels(
            text: str,
            allowed_labels: Sequence[
                str
            ],
            max_labels: int = 3,
    ) -> list[str]:
        """按出现顺序提取合法标签，并去重和截断。"""

        if max_labels < 1:
            raise ValueError(
                "max_labels不能小于1。"
            )

        if not isinstance(
                text,
                str,
        ):
            return []

        normalized_text = (
            text.strip()
        )

        if not normalized_text:
            return []

        normalized_labels: list[
            str
        ] = []

        seen_allowed_labels: set[
            str
        ] = set()

        for label in allowed_labels:
            normalized_label = (
                label.strip().upper()
            )

            if (
                not normalized_label
                or normalized_label
                in seen_allowed_labels
            ):
                continue

            seen_allowed_labels.add(
                normalized_label
            )

            normalized_labels.append(
                normalized_label
            )

        if not normalized_labels:
            return []

        # 长标签优先，
        # 防止短标签被长标签的一部分误匹配。
        pattern_labels = sorted(
            normalized_labels,

            key=len,

            reverse=True,
        )

        pattern = re.compile(
            (
                r"(?<![A-Z0-9_])(?:"

                + "|".join(
                    re.escape(
                        label
                    )

                    for label
                    in pattern_labels
                )

                + r")(?![A-Z0-9_])"
            ),

            flags=re.IGNORECASE,
        )

        selected_labels: list[
            str
        ] = []

        seen_selected_labels: set[
            str
        ] = set()

        for match in pattern.finditer(
            normalized_text
        ):
            selected_label = (
                match.group(0)
                .upper()
            )

            if (
                selected_label
                in seen_selected_labels
            ):
                continue

            seen_selected_labels.add(
                selected_label
            )

            selected_labels.append(
                selected_label
            )

        # NO_TOOL表示完全不需要业务工具。
        #
        # 如果模型同时返回其他工具组，
        # 以真实工具组为准，删除NO_TOOL。
        if (
            "NO_TOOL"
            in selected_labels

            and len(
                selected_labels
            ) > 1
        ):
            selected_labels = [
                label

                for label
                in selected_labels

                if label != "NO_TOOL"
            ]

        # 模型即使返回六七个标签，
        # 最终也只采用最靠前的三个。
        return selected_labels[
            :max_labels
        ]

    @staticmethod
    def _extract_first_router_label(
            text: str,
            allowed_labels: Sequence[
                str
            ],
    ) -> str | None:
        """从模型输出中提取第一个合法标签。"""

        selected_labels = (
            RetrievalModelManager
            ._extract_router_labels(
                text=text,

                allowed_labels=(
                    allowed_labels
                ),

                max_labels=1,
            )
        )

        if not selected_labels:
            return None

        return selected_labels[0]

    @staticmethod
    def _extract_router_response_text(
            response,
    ) -> str:
        """从llama.cpp聊天响应中读取文字内容。"""

        if not isinstance(
                response,
                dict,
        ):
            return ""

        choices = response.get(
            "choices"
        )

        if not isinstance(
                choices,
                list,
        ) or not choices:
            return ""

        first_choice = choices[0]

        if not isinstance(
                first_choice,
                dict,
        ):
            return ""

        message = first_choice.get(
            "message"
        )

        if not isinstance(
                message,
                dict,
        ):
            return ""

        content = message.get(
            "content"
        )

        if not isinstance(
                content,
                str,
        ):
            return ""

        return content.strip()

    @staticmethod
    def _extract_router_labels_after_think(
            text: str,
            allowed_labels: Sequence[
                str
            ],
            max_labels: int = 3,
    ) -> list[str]:
        """只从思考结束后的最终回答中提取合法标签。"""

        if not isinstance(
                text,
                str,
        ):
            return []

        normalized_text = (
            text.strip()
        )

        if not normalized_text:
            return []

        think_end_tag = (
            "</think>"
        )

        think_end_index = (
            normalized_text.rfind(
                think_end_tag
            )
        )

        if think_end_index < 0:
            return []

        final_answer = (
            normalized_text[
                think_end_index
                + len(
                    think_end_tag
                ):
            ]
            .strip()
        )

        if not final_answer:
            return []

        return (
            RetrievalModelManager
            ._extract_router_labels(
                text=(
                    final_answer
                ),

                allowed_labels=(
                    allowed_labels
                ),

                max_labels=(
                    max_labels
                ),
            )
        )

    @staticmethod
    def _extract_router_label_after_think(
            text: str,
            allowed_labels: Sequence[
                str
            ],
    ) -> str | None:
        """只从思考结束后的最终回答中提取第一个标签。"""

        selected_labels = (
            RetrievalModelManager
            ._extract_router_labels_after_think(
                text=text,

                allowed_labels=(
                    allowed_labels
                ),

                max_labels=1,
            )
        )

        if not selected_labels:
            return None

        return selected_labels[0]

    def _classify_labels_with_router(
            self,
            prompt: str,
            allowed_labels: Sequence[
                str
            ],
            trace_name: str,
            thinking: bool,
            max_tokens: int | None,
            max_labels: int,
    ) -> list[str] | None:
        """执行一次本地Router推理并返回合法标签列表。"""

        if not self.router_enabled:
            return None

        router_model = (
            self._router_model
        )

        if router_model is None:
            return None

        normalized_prompt = (
            prompt.strip()
        )

        if not normalized_prompt:
            return None

        if max_labels < 1:
            raise ValueError(
                "max_labels不能小于1。"
            )

        resolved_labels: list[
            str
        ] = []

        seen_labels: set[
            str
        ] = set()

        for label in allowed_labels:
            normalized_label = (
                label.strip()
                .upper()
            )

            if (
                not normalized_label
                or normalized_label
                in seen_labels
            ):
                continue

            seen_labels.add(
                normalized_label
            )

            resolved_labels.append(
                normalized_label
            )

        if not resolved_labels:
            raise ValueError(
                "allowed_labels不能为空。"
            )

        resolved_max_tokens = (
            self.router_max_tokens

            if max_tokens is None

            else max_tokens
        )

        if resolved_max_tokens < 1:
            raise ValueError(
                "max_tokens不能小于1。"
            )

        thinking_command = (
            "/think"

            if thinking

            else "/no_think"
        )

        router_prompt = (
            f"{normalized_prompt}\n\n"
            f"{thinking_command}"
        )

        try:
            with trace_span(
                    "local_router",

                    kind="llm",

                    input_value={
                        "task_name": (
                            trace_name
                        ),

                        "model_repo": (
                            self.router_model_repo
                        ),

                        "model_filename": (
                            self.router_model_filename
                        ),

                        "prompt": (
                            router_prompt
                        ),

                        "allowed_labels": (
                            resolved_labels
                        ),

                        "max_labels": (
                            max_labels
                        ),

                        "thinking_enabled": (
                            thinking
                        ),

                        "max_tokens": (
                            resolved_max_tokens
                        ),
                    },

                    attributes={
                        "local_model.type": (
                            "router"
                        ),

                        "local_model.runtime": (
                            "llama_cpp"
                        ),

                        "local_model.name": (
                            self.router_model_filename
                        ),

                        "router.task": (
                            trace_name
                        ),

                        "router.context_length": (
                            self.router_context_length
                        ),

                        "router.max_tokens": (
                            resolved_max_tokens
                        ),

                        "router.max_labels": (
                            max_labels
                        ),

                        "router.thinking_enabled": (
                            thinking
                        ),

                        "router.allowed_label_count": len(
                            resolved_labels
                        ),

                        "router.n_gpu_layers": -1,
                    },
            ) as span:

                with self._router_lock:
                    response = (
                        router_model
                        .create_chat_completion(
                            messages=[
                                {
                                    "role": "user",

                                    "content": (
                                        router_prompt
                                    ),
                                }
                            ],

                            max_tokens=(
                                resolved_max_tokens
                            ),

                            temperature=0.0,

                            top_p=1.0,

                            top_k=1,

                            stream=False,
                        )
                    )

                raw_output = (
                    self
                    ._extract_router_response_text(
                        response
                    )
                )

                found_complete_think_end = (
                    (
                        "</think>"
                        in raw_output
                    )

                    if thinking

                    else None
                )

                if thinking:
                    selected_labels = (
                        self
                        ._extract_router_labels_after_think(
                            text=(
                                raw_output
                            ),

                            allowed_labels=(
                                resolved_labels
                            ),

                            max_labels=(
                                max_labels
                            ),
                        )
                    )

                else:
                    selected_labels = (
                        self
                        ._extract_router_labels(
                            text=(
                                raw_output
                            ),

                            allowed_labels=(
                                resolved_labels
                            ),

                            max_labels=(
                                max_labels
                            ),
                        )
                    )

                response_usage = None

                if isinstance(
                    response,
                    dict,
                ):
                    response_usage = (
                        response.get(
                            "usage"
                        )
                    )

                valid_output = bool(
                    selected_labels
                )

                router_attributes = {
                    "router.output_chars": len(
                        raw_output
                    ),

                    "router.valid_output": (
                        valid_output
                    ),

                    "router.selected_label_count": len(
                        selected_labels
                    ),

                    "router.selected_labels": (
                        selected_labels
                    ),

                    "router.fallback_required": (
                        not valid_output
                    ),
                }

                if thinking:
                    router_attributes[
                        "router.complete_think_end"
                    ] = bool(
                        found_complete_think_end
                    )

                set_span_attributes(
                    span,

                    **router_attributes,
                )

                set_span_output(
                    span,

                    {
                        "status": (
                            "success"

                            if valid_output

                            else "invalid_output"
                        ),

                        "raw_output": (
                            raw_output
                        ),

                        "selected_labels": (
                            selected_labels
                        ),

                        "found_complete_think_end": (
                            found_complete_think_end
                        ),

                        "fallback_required": (
                            not valid_output
                        ),

                        "usage": (
                            response_usage
                        ),
                    },
                )

                if not selected_labels:
                    return None

                return selected_labels

        except Exception:
            logger.exception(
                "本地Router推理失败，"
                "本次将按上层策略降级"
            )

            return None

    def classify_with_router(
            self,
            prompt: str,
            allowed_labels: Sequence[
                str
            ],
            trace_name: str = (
                "Local Router"
            ),
            thinking: bool = False,
            max_tokens: int | None = None,
    ) -> str | None:
        """让本地小模型从合法标签中选择一个。"""

        selected_labels = (
            self._classify_labels_with_router(
                prompt=prompt,

                allowed_labels=(
                    allowed_labels
                ),

                trace_name=(
                    trace_name
                ),

                thinking=(
                    thinking
                ),

                max_tokens=(
                    max_tokens
                ),

                max_labels=1,
            )
        )

        if not selected_labels:
            return None

        return selected_labels[0]

    def classify_many_with_router(
            self,
            prompt: str,
            allowed_labels: Sequence[
                str
            ],
            trace_name: str = (
                "Local Router"
            ),
            thinking: bool = False,
            max_tokens: int | None = None,
            max_labels: int = 3,
    ) -> list[str] | None:
        """让本地小模型按重要程度选择多个合法标签。"""

        return self._classify_labels_with_router(
            prompt=prompt,

            allowed_labels=(
                allowed_labels
            ),

            trace_name=(
                trace_name
            ),

            thinking=(
                thinking
            ),

            max_tokens=(
                max_tokens
            ),

            max_labels=(
                max_labels
            ),
        )

    async def aclassify_with_router(
            self,
            prompt: str,
            allowed_labels: Sequence[
                str
            ],
            trace_name: str = (
                "Local Router"
            ),
            thinking: bool = False,
            max_tokens: int | None = None,
    ) -> str | None:
        """在线程中执行单标签本地路由推理。"""

        return await asyncio.to_thread(
            self.classify_with_router,

            prompt=(
                prompt
            ),

            allowed_labels=(
                allowed_labels
            ),

            trace_name=(
                trace_name
            ),

            thinking=(
                thinking
            ),

            max_tokens=(
                max_tokens
            ),
        )

    async def aclassify_many_with_router(
            self,
            prompt: str,
            allowed_labels: Sequence[
                str
            ],
            trace_name: str = (
                "Local Router"
            ),
            thinking: bool = False,
            max_tokens: int | None = None,
            max_labels: int = 3,
    ) -> list[str] | None:
        """在线程中执行多标签本地路由推理。"""

        return await asyncio.to_thread(
            self.classify_many_with_router,

            prompt=(
                prompt
            ),

            allowed_labels=(
                allowed_labels
            ),

            trace_name=(
                trace_name
            ),

            thinking=(
                thinking
            ),

            max_tokens=(
                max_tokens
            ),

            max_labels=(
                max_labels
            ),
        )


    def _require_embedding_model(
        self,
    ) -> SentenceTransformer:
        if self._embedding_model is None:
            raise RuntimeError(
                "Embedding模型尚未加载。"
            )

        return self._embedding_model

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """为多段文本生成归一化向量，并记录本地Embedding推理。"""

        resolved_texts = list(
            texts
        )

        if not resolved_texts:
            return []

        model = (
            self._require_embedding_model()
        )

        trace_input_items = [
            {
                "index": (
                    text_index
                ),

                "text_chars": len(
                    text
                ),

                "text_preview": (
                    _compact_local_model_trace_text(
                        text
                    )
                ),
            }

            for (
                text_index,
                text,
            ) in enumerate(
                resolved_texts
            )
        ]

        with trace_span(
                "local_embedding.encode",

                # 这里是真正执行向量模型推理的物理节点。
                kind="embedding",

                input_value={
                    "model_name": (
                        self.embedding_model_name
                    ),

                    "device": (
                        self.device
                    ),

                    "input_count": len(
                        resolved_texts
                    ),

                    "batch_size": (
                        self.embedding_batch_size
                    ),

                    "max_length": (
                        self.embedding_max_length
                    ),

                    "normalize_embeddings": (
                        True
                    ),

                    "texts": (
                        trace_input_items
                    ),
                },

                attributes={
                    "local_model.type": (
                        "embedding"
                    ),

                    "local_model.runtime": (
                        "sentence_transformers"
                    ),

                    "local_model.name": (
                        self.embedding_model_name
                    ),

                    "local_model.device": (
                        self.device
                    ),

                    "embedding.input_count": len(
                        resolved_texts
                    ),

                    "embedding.batch_size": (
                        self.embedding_batch_size
                    ),

                    "embedding.max_length": (
                        self.embedding_max_length
                    ),

                    "embedding.normalize": (
                        True
                    ),
                },
        ) as span:

            # SentenceTransformer模型实例
            # 不应同时被多个线程调用。
            with self._embedding_lock:
                vectors = model.encode(
                    resolved_texts,

                    batch_size=(
                        self.embedding_batch_size
                    ),

                    show_progress_bar=False,

                    convert_to_numpy=True,

                    normalize_embeddings=True,
                )

            float_vectors = (
                vectors
                .astype(
                    "float32",
                    copy=False,
                )
            )

            vector_shape = [
                int(
                    dimension
                )

                for dimension
                in float_vectors.shape
            ]

            vector_count = (
                int(
                    float_vectors.shape[0]
                )

                if float_vectors.ndim
                >= 1

                else 0
            )

            vector_dimensions = (
                int(
                    float_vectors.shape[-1]
                )

                if float_vectors.ndim
                >= 2

                else 0
            )

            set_span_attributes(
                span,

                **{
                    "embedding.output_count": (
                        vector_count
                    ),

                    "embedding.dimensions": (
                        vector_dimensions
                    ),

                    "embedding.output_dtype": (
                        str(
                            float_vectors.dtype
                        )
                    ),
                },
            )

            set_span_output(
                span,

                {
                    "status": (
                        "success"
                    ),

                    "input_count": len(
                        resolved_texts
                    ),

                    "vector_count": (
                        vector_count
                    ),

                    "vector_dimensions": (
                        vector_dimensions
                    ),

                    "vector_shape": (
                        vector_shape
                    ),

                    "dtype": str(
                        float_vectors.dtype
                    ),

                    "normalized": (
                        True
                    ),
                },
            )

            return (
                float_vectors
                .tolist()
            )
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """为单条检索请求生成向量。"""

        vectors = self.embed_documents(
            [text]
        )

        return vectors[0]
    async def aembed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return await asyncio.to_thread(
            self.embed_documents,
            texts,
        )

    async def aembed_query(
        self,
        text: str,
    ) -> list[float]:
        return await asyncio.to_thread(
            self.embed_query,
            text,
        )
    def _require_reranker_model(
        self,
    ) -> CrossEncoder:
        if self._reranker_model is None:
            raise RuntimeError(
                "Reranker模型尚未加载。"
            )

        return self._reranker_model
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int = 2,
    ) -> list[RerankResult]:
        """使用Cross-Encoder对候选文本进行精排并记录物理推理。"""

        resolved_documents = list(
            documents
        )

        if not resolved_documents:
            return []

        if top_k < 1:
            raise ValueError(
                "top_k不能小于1。"
            )

        model = (
            self._require_reranker_model()
        )

        resolved_top_k = min(
            top_k,
            len(
                resolved_documents
            ),
        )

        pairs = [
            (
                query,
                document,
            )

            for document
            in resolved_documents
        ]

        trace_document_items = [
            {
                "index": (
                    document_index
                ),

                "document_chars": len(
                    document
                ),

                "document_preview": (
                    _compact_local_model_trace_text(
                        document
                    )
                ),
            }

            for (
                document_index,
                document,
            ) in enumerate(
                resolved_documents
            )
        ]

        with trace_span(
                "local_cross_encoder.predict",

                # 这里是真正执行Cross-Encoder
                # Pair评分的物理模型节点。
                kind="reranker",

                input_value={
                    "model_name": (
                        self.reranker_model_name
                    ),

                    "device": (
                        self.device
                    ),

                    "query": {
                        "chars": len(
                            query
                        ),

                        "preview": (
                            _compact_local_model_trace_text(
                                query
                            )
                        ),
                    },

                    "documents": (
                        trace_document_items
                    ),

                    "pair_count": len(
                        pairs
                    ),

                    "requested_top_k": (
                        top_k
                    ),

                    "resolved_top_k": (
                        resolved_top_k
                    ),

                    "batch_size": (
                        self.reranker_batch_size
                    ),

                    "max_length": (
                        self.reranker_max_length
                    ),
                },

                attributes={
                    "local_model.type": (
                        "cross_encoder"
                    ),

                    "local_model.runtime": (
                        "sentence_transformers"
                    ),

                    "local_model.name": (
                        self.reranker_model_name
                    ),

                    "local_model.device": (
                        self.device
                    ),

                    "reranker.input_count": len(
                        resolved_documents
                    ),

                    "reranker.pair_count": len(
                        pairs
                    ),

                    "reranker.requested_top_k": (
                        top_k
                    ),

                    "reranker.resolved_top_k": (
                        resolved_top_k
                    ),

                    "reranker.batch_size": (
                        self.reranker_batch_size
                    ),

                    "reranker.max_length": (
                        self.reranker_max_length
                    ),
                },
        ) as span:

            # CrossEncoder模型实例
            # 不应同时被多个线程调用。
            with self._reranker_lock:
                scores = model.predict(
                    pairs,

                    batch_size=(
                        self.reranker_batch_size
                    ),

                    show_progress_bar=False,

                    convert_to_numpy=True,
                )

            flat_scores = (
                scores
                .reshape(-1)
                .tolist()
            )

            if (
                len(
                    flat_scores
                )
                != len(
                    resolved_documents
                )
            ):
                raise RuntimeError(
                    "Cross-Encoder返回的分数数量"
                    "与候选文档数量不一致："
                    f"scores={len(flat_scores)}，"
                    f"documents={len(resolved_documents)}。"
                )

            results = [
                RerankResult(
                    index=index,

                    text=document,

                    score=float(
                        flat_scores[
                            index
                        ]
                    ),
                )

                for (
                    index,
                    document,
                ) in enumerate(
                    resolved_documents
                )
            ]

            results.sort(
                key=lambda item: (
                    item.score
                ),

                reverse=True,
            )

            selected_results = (
                results[
                    :resolved_top_k
                ]
            )

            raw_score_items = [
                {
                    "index": (
                        score_index
                    ),

                    "score": round(
                        float(
                            score
                        ),
                        8,
                    ),
                }

                for (
                    score_index,
                    score,
                ) in enumerate(
                    flat_scores
                )
            ]

            ranked_result_items = [
                {
                    "rank": (
                        rank
                    ),

                    "index": (
                        result.index
                    ),

                    "score": round(
                        result.score,
                        8,
                    ),

                    "text_preview": (
                        _compact_local_model_trace_text(
                            result.text
                        )
                    ),
                }

                for (
                    rank,
                    result,
                ) in enumerate(
                    selected_results,
                    start=1,
                )
            ]

            best_score = (
                selected_results[0].score

                if selected_results

                else None
            )

            set_span_attributes(
                span,

                **{
                    "reranker.output_count": len(
                        selected_results
                    ),

                    "reranker.best_score": (
                        best_score
                        if best_score
                        is not None
                        else "null"
                    ),
                },
            )

            set_span_output(
                span,

                {
                    "status": (
                        "success"
                    ),

                    "pair_count": len(
                        pairs
                    ),

                    # 按原始候选顺序保存物理模型分数。
                    "raw_scores": (
                        raw_score_items
                    ),

                    # 按最终分数从高到低保存Top-K。
                    "ranked_results": (
                        ranked_result_items
                    ),

                    "returned_count": len(
                        selected_results
                    ),
                },
            )

            return selected_results
    async def arerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int = 2,
    ) -> list[RerankResult]:
        return await asyncio.to_thread(
            self.rerank,
            query,
            documents,
            top_k,
        )

    def close(
            self,
    ) -> None:
        """释放本地模型占用的内存和显存。"""

        with self._embedding_lock:
            self._embedding_model = None

        with self._reranker_lock:
            self._reranker_model = None

        with self._router_lock:
            router_model = (
                self._router_model
            )

            self._router_model = None

            if router_model is not None:
                try:
                    router_model.close()

                except Exception:
                    logger.exception(
                        "释放本地Memory Router时发生异常"
                    )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "本地检索模型和Memory Router已释放。"
        )

    async def aclose(
        self,
    ) -> None:
        await asyncio.to_thread(
            self.close
        )