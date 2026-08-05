from __future__ import annotations

from dataclasses import (
    dataclass,
)
from threading import (
    RLock,
)
from typing import (
    Any,
)

import networkx as nx


@dataclass(frozen=True)
class GraphMemoryHit:
    """表示通过图多跳扩展找到的一条记忆。"""

    memory_id: str
    distance: int


class MemoryGraphIndex:
    """使用NetworkX维护active长期记忆的辅助图索引。

    SQLite Memory Store仍然是唯一真相源。

    这个图只负责：
    1. 保存active记忆中的实体关系；
    2. 从向量召回的种子记忆向外多跳扩展；
    3. 根据memory_id删除已经retired的图边；
    4. 对完全相同的记忆执行确定性去重。
    """

    def __init__(
        self,
    ) -> None:
        # 记忆关系具有方向，而且相同节点之间
        # 可能同时存在多条来自不同记忆的关系。
        self.graph = nx.MultiDiGraph()

        # memory_id对应它在图中的全部边：
        #
        # {
        #     memory_id: {
        #         (source, target, edge_key),
        #     }
        # }
        self._memory_edges: dict[
            str,
            set[
                tuple[
                    str,
                    str,
                    str,
                ]
            ],
        ] = {}

        # NetworkX本身不承诺线程安全。
        # 当前图操作虽然非常短，但加锁后更稳妥。
        self._lock = RLock()

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:
        """压缩空白并生成便于比较的文本。"""

        if not isinstance(
            value,
            str,
        ):
            return ""

        return " ".join(
            value
            .strip()
            .split()
        )

    @classmethod
    def _entity_key(
        cls,
        value: Any,
    ) -> str:
        """把实体文本转换成稳定节点Key。"""

        normalized = cls._normalize_text(
            value
        )

        return normalized.casefold()

    @classmethod
    def _relation_key(
        cls,
        value: Any,
    ) -> str:
        """把关系转换成稳定的snake_case形式。"""

        normalized = cls._normalize_text(
            value
        )

        normalized = (
            normalized
            .replace(
                "-",
                "_",
            )
            .replace(
                " ",
                "_",
            )
            .casefold()
        )

        while "__" in normalized:
            normalized = normalized.replace(
                "__",
                "_",
            )

        return normalized.strip(
            "_"
        )

    @classmethod
    def _triple_signature(
        cls,
        triple: dict[
            str,
            Any,
        ],
    ) -> tuple[
        str,
        str,
        str,
    ] | None:
        """生成用于判断图关系是否相同的规范化签名。"""

        subject_key = cls._entity_key(
            triple.get(
                "subject"
            )
        )

        relation_key = cls._relation_key(
            triple.get(
                "relation"
            )
        )

        object_key = cls._entity_key(
            triple.get(
                "object"
            )
        )

        if not (
            subject_key
            and relation_key
            and object_key
        ):
            return None

        return (
            subject_key,
            relation_key,
            object_key,
        )

    def clear(
        self,
    ) -> None:
        """清空整个图索引。"""

        with self._lock:
            self.graph.clear()
            self._memory_edges.clear()

    def add_memory(
        self,
        memory_id: str,
        value: dict[
            str,
            Any,
        ],
    ) -> int:
        """把一条active记忆加入图中。

        返回实际添加的图边数量。
        """

        if not isinstance(
            memory_id,
            str,
        ) or not memory_id:
            return 0

        if not isinstance(
            value,
            dict,
        ):
            return 0

        if value.get(
            "status"
        ) != "active":
            return 0

        triples = value.get(
            "triples",
            [],
        )

        if not isinstance(
            triples,
            list,
        ):
            return 0

        with self._lock:
            # 允许相同memory_id被安全地重复同步。
            self.remove_memory(
                memory_id
            )

            edge_references: set[
                tuple[
                    str,
                    str,
                    str,
                ]
            ] = set()

            seen_signatures: set[
                tuple[
                    str,
                    str,
                    str,
                ]
            ] = set()

            for triple_index, triple in enumerate(
                triples
            ):
                if not isinstance(
                    triple,
                    dict,
                ):
                    continue

                signature = (
                    self._triple_signature(
                        triple
                    )
                )

                if signature is None:
                    continue

                (
                    subject_key,
                    relation_key,
                    object_key,
                ) = signature

                # “用户 relates_to 用户”这样的自环
                # 对当前多跳召回没有实际帮助。
                if subject_key == object_key:
                    continue

                # 同一条记忆内部重复输出相同三元组时，
                # 只保留一条图边。
                if signature in seen_signatures:
                    continue

                seen_signatures.add(
                    signature
                )

                subject_label = (
                    self._normalize_text(
                        triple.get(
                            "subject"
                        )
                    )
                )

                object_label = (
                    self._normalize_text(
                        triple.get(
                            "object"
                        )
                    )
                )

                subject_type = (
                    self._normalize_text(
                        triple.get(
                            "subject_type"
                        )
                    )
                    or "entity"
                )

                object_type = (
                    self._normalize_text(
                        triple.get(
                            "object_type"
                        )
                    )
                    or "entity"
                )

                self.graph.add_node(
                    subject_key,

                    label=subject_label,

                    node_type=(
                        subject_type
                    ),
                )

                self.graph.add_node(
                    object_key,

                    label=object_label,

                    node_type=(
                        object_type
                    ),
                )

                edge_key = (
                    f"{memory_id}:"
                    f"{triple_index}"
                )

                self.graph.add_edge(
                    subject_key,
                    object_key,

                    key=edge_key,

                    memory_id=(
                        memory_id
                    ),

                    relation=(
                        relation_key
                    ),

                    content=(
                        value.get(
                            "content",
                            "",
                        )
                    ),

                    memory_type=(
                        value.get(
                            "memory_type",
                            "general",
                        )
                    ),

                    importance=(
                        value.get(
                            "importance",
                            2,
                        )
                    ),

                    created_at=(
                        value.get(
                            "created_at",
                            "",
                        )
                    ),

                    updated_at=(
                        value.get(
                            "updated_at",
                            "",
                        )
                    ),
                    valid_from=(
                        value.get(
                            "valid_from"
                        )
                    ),

                    expires_at=(
                        value.get(
                            "expires_at"
                        )
                    ),
                )

                edge_references.add(
                    (
                        subject_key,
                        object_key,
                        edge_key,
                    )
                )

            if edge_references:
                self._memory_edges[
                    memory_id
                ] = edge_references

            return len(
                edge_references
            )

    def remove_memory(
        self,
        memory_id: str,
    ) -> int:
        """删除某条记忆在图中的所有边。"""

        with self._lock:
            edge_references = (
                self._memory_edges
                .pop(
                    memory_id,
                    None,
                )
            )

            # 正常情况下直接使用反向索引。
            #
            # 如果反向索引因异常缺失，
            # 再扫描一次图，保证删除操作仍然可靠。
            if edge_references is None:
                edge_references = set()

                for (
                    source,
                    target,
                    edge_key,
                    edge_data,
                ) in self.graph.edges(
                    keys=True,
                    data=True,
                ):
                    if (
                        edge_data.get(
                            "memory_id"
                        )
                        == memory_id
                    ):
                        edge_references.add(
                            (
                                source,
                                target,
                                edge_key,
                            )
                        )

            removed_count = 0

            affected_nodes: set[
                str
            ] = set()

            for (
                source,
                target,
                edge_key,
            ) in edge_references:
                if not self.graph.has_edge(
                    source,
                    target,
                    edge_key,
                ):
                    continue

                self.graph.remove_edge(
                    source,
                    target,
                    edge_key,
                )

                removed_count += 1

                affected_nodes.add(
                    source
                )

                affected_nodes.add(
                    target
                )

            # 删除已经没有任何边的孤立节点，
            # 防止图里长期残留无效实体。
            for node in affected_nodes:
                if (
                    node in self.graph
                    and self.graph.degree(
                        node
                    ) == 0
                ):
                    self.graph.remove_node(
                        node
                    )

            return removed_count

    def find_duplicate_memory_ids(
            self,
            content: str,
            triples: list[
                dict[
                    str,
                    Any,
                ]
            ],
            valid_from: str | None = None,
            expires_at: str | None = None,
    ) -> list[str]:
        """查找内容和三元组均完全相同的active记忆。

        这里只处理确定性的完全重复，
        不尝试判断语义冲突。
        """

        normalized_content = (
            self._normalize_text(
                content
            )
            .casefold()
        )

        if not normalized_content:
            return []

        candidate_signatures = {
            signature
            for triple in triples
            if (
                signature
                := self._triple_signature(
                    triple
                )
            )
            is not None
        }

        if not candidate_signatures:
            return []

        duplicate_ids: set[
            str
        ] = set()

        with self._lock:
            for (
                source,
                target,
                _edge_key,
                edge_data,
            ) in self.graph.edges(
                keys=True,
                data=True,
            ):
                relation = (
                    self._relation_key(
                        edge_data.get(
                            "relation"
                        )
                    )
                )

                edge_signature = (
                    source,
                    relation,
                    target,
                )

                if (
                    edge_signature
                    not in candidate_signatures
                ):
                    continue
                if (
                    edge_data.get(
                        "valid_from"
                    )
                    != valid_from
                ):
                    continue

                if (
                    edge_data.get(
                        "expires_at"
                    )
                    != expires_at
                ):
                    continue
                edge_content = (
                    self._normalize_text(
                        edge_data.get(
                            "content"
                        )
                    )
                    .casefold()
                )

                if (
                    edge_content
                    != normalized_content
                ):
                    continue

                memory_id = edge_data.get(
                    "memory_id"
                )

                if isinstance(
                    memory_id,
                    str,
                ):
                    duplicate_ids.add(
                        memory_id
                    )

        return sorted(
            duplicate_ids
        )

    def expand_from_memory_ids(
        self,
        seed_memory_ids: list[str],
        max_hops: int = 2,
        limit: int = 6,
    ) -> list[GraphMemoryHit]:
        """从种子记忆涉及的节点向外执行多跳扩展。"""

        if max_hops < 1:
            return []

        if limit < 1:
            return []

        normalized_seed_ids = {
            memory_id
            for memory_id
            in seed_memory_ids
            if isinstance(
                memory_id,
                str,
            ) and memory_id
        }

        if not normalized_seed_ids:
            return []

        with self._lock:
            seed_nodes: set[
                str
            ] = set()

            for memory_id in (
                normalized_seed_ids
            ):
                for (
                    source,
                    target,
                    _edge_key,
                ) in self._memory_edges.get(
                    memory_id,
                    set(),
                ):
                    seed_nodes.add(
                        source
                    )

                    seed_nodes.add(
                        target
                    )

            if not seed_nodes:
                return []

            # 召回时同时允许沿正向边和反向边扩展。
            #
            # 图中仍保留原始方向，
            # 这里只使用无向视图执行关联搜索。
            traversal_graph = (
                self.graph
                .to_undirected(
                    as_view=True
                )
            )

            node_distances: dict[
                str,
                int,
            ] = {}

            for seed_node in seed_nodes:
                lengths = (
                    nx
                    .single_source_shortest_path_length(
                        traversal_graph,

                        source=(
                            seed_node
                        ),

                        cutoff=(
                            max_hops
                        ),
                    )
                )

                for node, distance in (
                    lengths.items()
                ):
                    current_distance = (
                        node_distances.get(
                            node
                        )
                    )

                    if (
                        current_distance is None
                        or distance
                        < current_distance
                    ):
                        node_distances[
                            node
                        ] = distance

            memory_distances: dict[
                str,
                int,
            ] = {}

            memory_importance: dict[
                str,
                int,
            ] = {}

            for (
                source,
                target,
                _edge_key,
                edge_data,
            ) in self.graph.edges(
                keys=True,
                data=True,
            ):
                if (
                    source
                    not in node_distances
                    or target
                    not in node_distances
                ):
                    continue

                memory_id = edge_data.get(
                    "memory_id"
                )

                if not isinstance(
                    memory_id,
                    str,
                ):
                    continue

                if memory_id in (
                    normalized_seed_ids
                ):
                    continue

                # 一条边所处的跳数，
                # 使用两个端点中更远的那个。
                distance = max(
                    node_distances[
                        source
                    ],

                    node_distances[
                        target
                    ],
                )

                if distance > max_hops:
                    continue

                previous_distance = (
                    memory_distances.get(
                        memory_id
                    )
                )

                if (
                    previous_distance is None
                    or distance
                    < previous_distance
                ):
                    memory_distances[
                        memory_id
                    ] = distance

                importance = edge_data.get(
                    "importance",
                    2,
                )

                if not isinstance(
                    importance,
                    int,
                ):
                    importance = 2

                memory_importance[
                    memory_id
                ] = max(
                    importance,

                    memory_importance.get(
                        memory_id,
                        1,
                    ),
                )

            ranked_memory_ids = sorted(
                memory_distances,

                key=lambda memory_id: (
                    memory_distances[
                        memory_id
                    ],

                    -memory_importance.get(
                        memory_id,
                        2,
                    ),

                    memory_id,
                ),
            )

            return [
                GraphMemoryHit(
                    memory_id=memory_id,

                    distance=(
                        memory_distances[
                            memory_id
                        ]
                    ),
                )
                for memory_id
                in ranked_memory_ids[
                    :limit
                ]
            ]

    def stats(
        self,
    ) -> dict[
        str,
        int,
    ]:
        """返回便于日志展示的图统计信息。"""

        with self._lock:
            return {
                "nodes": (
                    self.graph
                    .number_of_nodes()
                ),

                "edges": (
                    self.graph
                    .number_of_edges()
                ),

                "memories": len(
                    self._memory_edges
                ),
            }