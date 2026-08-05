from __future__ import annotations

from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    Sequence,
)


# Router的保留标签。
# 它表示本轮不需要任何业务工具，
# 本身不是一个真实Toolset。
NO_TOOL_ROUTE = "NO_TOOL"




@dataclass(frozen=True)
class ToolsetSpec:
    """描述一个能力域及其完整工具包。"""

    name: str
    description: str
    instructions: str

    required_tool_names: tuple[
        str,
        ...,
    ]

    optional_tool_names: tuple[
        str,
        ...,
    ] = ()

@dataclass(frozen=True)
class ToolsetResolution:
    """表示Toolset在当前真实工具池中的展开结果。"""

    spec: ToolsetSpec

    tools: tuple[
        Any,
        ...,
    ]

    missing_required_tool_names: tuple[
        str,
        ...,
    ]

    missing_optional_tool_names: tuple[
        str,
        ...,
    ]

    @property
    def is_available(
        self,
    ) -> bool:
        """必需工具全部存在时才允许激活。"""

        return not (
            self.missing_required_tool_names
        )

    @property
    def tool_names(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        """返回最终展开出的工具名称。"""

        return tuple(
            _read_tool_name(
                tool
            )

            for tool in self.tools
        )


class ToolsetRegistry:
    """保存Toolset定义，并确定性展开真实工具。"""

    def __init__(
        self,
        specs: Sequence[
            ToolsetSpec
        ],

    ) -> None:



        specs_by_name: dict[
            str,
            ToolsetSpec,
        ] = {}

        for spec in specs:
            self._validate_spec(
                spec
            )

            if spec.name in specs_by_name:
                raise ValueError(
                    "Toolset名称重复："
                    f"{spec.name}"
                )

            specs_by_name[
                spec.name
            ] = spec

        if not specs_by_name:
            raise ValueError(
                "ToolsetRegistry至少需要"
                "一个Toolset。"
            )

        self._specs_by_name = (
            specs_by_name
        )


    @staticmethod
    def _validate_spec(
        spec: ToolsetSpec,
    ) -> None:
        """检查一个Toolset定义是否完整。"""

        if not spec.name:
            raise ValueError(
                "Toolset名称不能为空。"
            )

        if spec.name != spec.name.upper():
            raise ValueError(
                "Toolset名称必须使用大写标签："
                f"{spec.name}"
            )

        if not spec.description.strip():
            raise ValueError(
                f"Toolset {spec.name}"
                "缺少description。"
            )

        if not spec.instructions.strip():
            raise ValueError(
                f"Toolset {spec.name}"
                "缺少instructions。"
            )


        required_names = set(
            spec.required_tool_names
        )

        optional_names = set(
            spec.optional_tool_names
        )

        if (
            len(required_names)
            != len(spec.required_tool_names)
        ):
            raise ValueError(
                f"Toolset {spec.name}的required工具"
                "存在重复名称。"
            )

        if (
            len(optional_names)
            != len(spec.optional_tool_names)
        ):
            raise ValueError(
                f"Toolset {spec.name}的optional工具"
                "存在重复名称。"
            )

        if "" in required_names | optional_names:
            raise ValueError(
                f"Toolset {spec.name}中"
                "存在空工具名称。"
            )

        duplicated_names = (
            required_names
            & optional_names
        )

        if duplicated_names:
            raise ValueError(
                f"Toolset {spec.name}中的工具"
                "不能同时属于required和optional："
                + ", ".join(
                    sorted(
                        duplicated_names
                    )
                )
            )


    @property
    def toolset_names(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        """返回全部真实Toolset标签。"""

        return tuple(
            self._specs_by_name
        )

    @property
    def defined_route_labels(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        """返回Registry中定义的全部路由标签。"""

        return (
            NO_TOOL_ROUTE,
            *self.toolset_names,
        )

    def available_route_labels(
        self,
        available_tools: Sequence[
            Any
        ],
    ) -> tuple[
        str,
        ...,
    ]:
        """返回当前真实工具池允许选择的标签。"""

        available_names = tuple(
            item["name"]

            for item
            in self.build_router_metadata(
                available_tools
            )
        )

        return (
            NO_TOOL_ROUTE,
            *available_names,
        )

    def get(
        self,
        toolset_name: str,
    ) -> ToolsetSpec | None:
        """按名称读取Toolset。"""

        return (
            self._specs_by_name
            .get(
                toolset_name
                .strip()
                .upper()
            )
        )

    def resolve(
        self,
        toolset_name: str,
        available_tools: Sequence[
            Any
        ],
    ) -> ToolsetResolution:
        """从当前真实工具池展开一个Toolset。"""

        spec = self.get(
            toolset_name
        )

        if spec is None:
            raise KeyError(
                "不存在Toolset："
                f"{toolset_name!r}"
            )

        tools_by_name = (
            _index_tools_by_name(
                available_tools
            )
        )

        missing_required_names = tuple(
            tool_name

            for tool_name
            in spec.required_tool_names

            if tool_name
            not in tools_by_name
        )

        missing_optional_names = tuple(
            tool_name

            for tool_name
            in spec.optional_tool_names

            if tool_name
            not in tools_by_name
        )

        selected_tools = tuple(
            tools_by_name[
                tool_name
            ]

            for tool_name in (
                *spec.required_tool_names,
                *spec.optional_tool_names,
            )

            if tool_name
            in tools_by_name
        )

        return ToolsetResolution(
            spec=spec,

            tools=selected_tools,

            missing_required_tool_names=(
                missing_required_names
            ),

            missing_optional_tool_names=(
                missing_optional_names
            ),
        )

    def build_router_metadata(
        self,
        available_tools: Sequence[
            Any
        ],
        *,
        extra_available_tool_names: Sequence[
            str
        ] = (),
    ) -> list[
        dict[
            str,
            str,
        ]
    ]:
        """生成当前真正可用的Toolset能力目录。

        available_tools：
            当前真实工具对象。

        extra_available_tool_names：
            由LangChain Middleware动态生成、
            因而不直接存在于基础工具列表中的能力名称。

        这里只用于判断Toolset是否可用，
        不会创建工具，也不会把占位对象交给模型执行。
        """

        tools_by_name = (
            _index_tools_by_name(
                available_tools
            )
        )

        available_tool_names = set(
            tools_by_name
        )

        for raw_tool_name in (
            extra_available_tool_names
        ):
            tool_name = str(
                raw_tool_name
            ).strip()

            if not tool_name:
                continue

            available_tool_names.add(
                tool_name
            )

        metadata: list[
            dict[
                str,
                str,
            ]
        ] = []

        for toolset_name in (
            self.toolset_names
        ):
            spec = self._specs_by_name[
                toolset_name
            ]

            missing_required_names = [
                tool_name

                for tool_name
                in spec.required_tool_names

                if tool_name
                not in available_tool_names
            ]

            if missing_required_names:
                continue

            metadata.append(
                {
                    "name": (
                        spec.name
                    ),

                    "description": (
                        spec.description
                    ),
                }
            )

        return metadata


def _read_tool_name(
    tool: Any,
) -> str:
    """读取LangChain Tool名称。"""

    return str(
        getattr(
            tool,
            "name",
            "",
        )
    ).strip()


def _index_tools_by_name(
    tools: Sequence[
        Any
    ],
) -> dict[
    str,
    Any,
]:
    """把真实工具池转换成名称索引。"""

    tools_by_name: dict[
        str,
        Any,
    ] = {}

    for tool in tools:
        tool_name = (
            _read_tool_name(
                tool
            )
        )

        if not tool_name:
            raise ValueError(
                "发现没有name属性的工具。"
            )

        if tool_name in tools_by_name:
            raise ValueError(
                "当前工具池存在重复名称："
                f"{tool_name}"
            )

        tools_by_name[
            tool_name
        ] = tool

    return tools_by_name


DEFAULT_TOOLSET_REGISTRY = (
    ToolsetRegistry(
        specs=[
            ToolsetSpec(
                name="WEB_RESEARCH",

                description=(
                    "搜索新闻、当前公开信息和网页内容；"
                    "不负责点击、填写或登录操作。"
                ),

                instructions=(
                    "涉及今天、最近或当前时先确认时间；"
                    "先用web_search发现来源，"
                    "摘要不足时再打开网页读取。"
                ),

                required_tool_names=(
                    "get_current_time",
                    "web_search",
                ),

                optional_tool_names=(
                    "browser_navigate",
                    "browser_snapshot",
                    "browser_find",
                    "browser_tabs",
                    "browser_close",
                ),
            ),

            ToolsetSpec(
                name="BROWSER_AUTOMATION",

                description=(
                    "执行真实浏览器导航、点击、输入、"
                    "表单填写和多页面交互。"
                ),

                instructions=(
                    "先导航并读取页面快照，"
                    "再根据真实页面结构执行交互；"
                    "不要猜测页面元素。"
                ),

                required_tool_names=(
                    "browser_navigate",
                    "browser_snapshot",
                    "browser_click",
                    "browser_type",
                    "browser_close",
                ),

                optional_tool_names=(
                    "browser_find",
                    "browser_fill_form",
                    "browser_select_option",
                    "browser_wait_for",
                    "browser_tabs",
                    "browser_navigate_back",
                ),
            ),

            ToolsetSpec(
                name="FILE_INSPECTION",

                description=(
                    "查找本地文件、浏览目录、"
                    "搜索代码或读取文本内容；"
                    "不修改文件。"
                ),

                instructions=(
                    "先定位目标文件，再按需读取相关片段；"
                    "只读任务不要调用写入工具。"
                ),

                required_tool_names=(
                    "list_directory",
                    "find_files",
                    "grep_files",
                    "read_file",
                ),
            ),

            ToolsetSpec(
                name="FILE_EDITING",

                description=(
                    "创建、覆盖或精确修改"
                    "本地文本文件。"
                ),

                instructions=(
                    "修改前先读取和定位；"
                    "优先使用精确替换；"
                    "修改后进行最小验证。"
                ),

                required_tool_names=(
                    "read_file",
                    "grep_files",
                    "replace_in_file",
                    "write_file",
                    "find_files",
                ),

                optional_tool_names=(
                    "list_directory",
                ),
            ),

            ToolsetSpec(
                name="SOFTWARE_DEVELOPMENT",

                description=(
                    "运行Python、Git、pytest、"
                    "语法检查或其他Shell调试任务。"
                ),

                instructions=(
                    "先读取相关代码，"
                    "再执行最小验证命令；"
                    "必须依据真实Shell输出判断结果。"
                ),

                required_tool_names=(
                    "read_file",
                    "grep_files",
                    "shell",
                ),

                optional_tool_names=(
                    "find_files",
                    "list_directory",
                    "replace_in_file",
                    "write_file",
                    "find_github_mirror",
                ),
            ),
        ]
    )
)