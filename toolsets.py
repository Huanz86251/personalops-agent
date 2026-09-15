from __future__ import annotations

from prompt_loader import load_prompt

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

# Deep Agents使用短工具名，旧主运行时使用同能力的历史名称。
# Registry在解析真实工具池时接受这些确定性别名；模型和Cross-Encoder
# 始终只看到实际存在的工具对象，不会得到虚构别名。
TOOL_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "list_directory": ("ls",),
    "find_files": ("glob",),
    "grep_files": ("grep",),
    "replace_in_file": ("edit_file",),
    "shell": ("execute",),
}


@dataclass(frozen=True)
class ToolsetSpec:
    """描述一个能力域及其完整工具包。"""

    name: str
    description: str
    routing_profile: str
    routing_threshold: float
    instructions: str

    required_tool_names: tuple[
        str,
        ...,
    ]

    optional_tool_names: tuple[
        str,
        ...,
    ] = ()

    # 互斥组代表一个完整、独立的执行环境。它成为最高分主组时，
    # 不能再把其他业务组同时暴露给执行模型；角色公共工具仍由
    # Middleware 在组外追加。
    exclusive: bool = False

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

        if not spec.routing_profile.strip():
            raise ValueError(
                f"Toolset {spec.name}缺少routing_profile。"
            )

        missing_routing_sections = [
            section
            for section in ("## 选择它", "## 典型命令")
            if section not in spec.routing_profile
        ]
        if missing_routing_sections:
            raise ValueError(
                f"Toolset {spec.name}的routing_profile缺少章节："
                + ", ".join(missing_routing_sections)
            )

        if not 0.0 <= spec.routing_threshold <= 1.0:
            raise ValueError(
                f"Toolset {spec.name}的routing_threshold必须在0到1之间。"
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

        def resolved_name(tool_name: str) -> str | None:
            if tool_name in tools_by_name:
                return tool_name
            return next(
                (
                    alias
                    for alias in TOOL_NAME_ALIASES.get(tool_name, ())
                    if alias in tools_by_name
                ),
                None,
            )

        required_resolutions = {
            name: resolved_name(name)
            for name in spec.required_tool_names
        }
        optional_resolutions = {
            name: resolved_name(name)
            for name in spec.optional_tool_names
        }

        missing_required_names = tuple(
            name
            for name, resolved in required_resolutions.items()
            if resolved is None
        )
        missing_optional_names = tuple(
            name
            for name, resolved in optional_resolutions.items()
            if resolved is None
        )

        selected_tools_list: list[Any] = []
        selected_actual_names: set[str] = set()
        for resolved in (
            *required_resolutions.values(),
            *optional_resolutions.values(),
        ):
            if resolved is None or resolved in selected_actual_names:
                continue
            selected_tools_list.append(tools_by_name[resolved])
            selected_actual_names.add(resolved)
        selected_tools = tuple(selected_tools_list)

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
                name="FEISHU_FILE_EXPORT",
                description="授权用户通过 General 申请将允许目录内的本地文件或目录发回当前飞书会话；上传前必须由用户单独确认，未配置时不可用。",
                routing_profile=load_prompt("routing/toolsets/feishu_file_export"),
                # “在飞书提醒我”也包含“飞书”，但不是文件回传。
                # Secondary候选还有0.05共享容差；实际文件回传样例约0.56，
                # 提醒样例约0.45，因此用0.52阻断后者并保留前者。
                routing_threshold=0.52,
                instructions="使用 send_local_file_to_feishu 提交文件或目录的绝对路径。PENDING 只能报告等待用户确认；不得通过 Shell、浏览器或其他工具绕过授权。",
                required_tool_names=("send_local_file_to_feishu",),
            ),
            ToolsetSpec(
                name="LOCAL_DOCUMENTS",
                description="读取PDF/PPTX等文档并自动OCR、独立中英文图片OCR、格式转换及Excel操作；无需MCP服务。",
                routing_profile=load_prompt("routing/toolsets/local_documents"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/local_documents"),
                required_tool_names=("attachment_to_text", "ocr_image", "convert_document", "spreadsheet_read", "spreadsheet_write", "spreadsheet_format", "spreadsheet_chart"),
            ),
            ToolsetSpec(
                name="LOCAL_ANALYSIS",
                description="本地SymPy代数、Python语法检查和Ruff静态检查，不执行提交的代码。",
                routing_profile=load_prompt("routing/toolsets/local_analysis"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/local_analysis"),
                required_tool_names=("symbolic_math", "python_syntax_check", "python_static_check"),
            ),
            ToolsetSpec(
                name="DESKTOP_OBSERVATION",
                description=(
                    "按用户明确要求截取当前Windows桌面，并可用本地OCR识别截图文字；"
                    "当前不包含点击、输入或持续屏幕监控。"
                ),
                routing_profile=load_prompt("routing/toolsets/desktop_observation"),
                routing_threshold=0.45,
                instructions=load_prompt("tools/desktop_observation"),
                required_tool_names=("capture_desktop_screenshot", "ocr_image"),
            ),
            ToolsetSpec(
                name="SCHEDULED_AUTOMATION",
                description=(
                    "创建、查询、暂停、恢复或取消持久化日程；可在到期时"
                    "弹出Windows通知、向当前飞书会话推送文本，或把一次性任务"
                    "作为新Event排入Agent队列；不负责外部日历和邮件。"
                ),
                routing_profile=load_prompt("routing/toolsets/scheduled_automation"),
                # Secondary candidates receive a small shared tolerance in
                # ToolsetRouter. Keep this group at 0.45 so unrelated
                # multi-tool tasks scoring around 0.36 cannot slip in, while
                # representative reminder intents remain above 0.56.
                routing_threshold=0.45,
                instructions=load_prompt("tools/scheduled_automation"),
                required_tool_names=(
                    "get_current_time",
                    "schedule_create",
                    "schedule_create_feishu_reminder",
                    "schedule_create_agent_task",
                    "schedule_list",
                    "schedule_pause",
                    "schedule_resume",
                    "schedule_delete",
                    "schedule_runs",
                    "windows_notify",
                ),
            ),
            ToolsetSpec(
                name="EMAIL_READING",
                description=(
                    "通过本地只读IMAP连接查看QQ、Foxmail或其他兼容邮箱的"
                    "连接状态、近期邮件、正文摘要和附件，并可将指定附件保存到"
                    "本机隔离目录；不能发送、删除、移动或标记邮件。"
                ),
                routing_profile=load_prompt("routing/toolsets/email_reading"),
                routing_threshold=0.40,
                instructions=load_prompt("tools/email_reading"),
                required_tool_names=(
                    "email_connection_status",
                    "email_list_recent",
                    "email_get_snippet",
                    "email_read_message",
                    "email_list_attachments",
                    "email_download_attachment",
                    "email_create_draft",
                ),
            ),
            ToolsetSpec(
                name="WEB_RESEARCH",

                description=(
                    "搜索新闻、当前公开信息和网页内容；"
                    "不负责点击、填写或登录操作。"
                ),

                routing_profile=load_prompt("routing/toolsets/web_research"),
                routing_threshold=0.35,

                instructions=(
                    load_prompt("tools/web_research")
                ),

                required_tool_names=(
                    "get_current_time",
                    "web_search",
                ),

                optional_tool_names=(
                    "fetch_webpage",
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

                routing_profile=load_prompt("routing/toolsets/browser_automation"),
                routing_threshold=0.35,

                instructions=(
                    load_prompt("tools/browser_automation")
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

                routing_profile=load_prompt("routing/toolsets/file_inspection"),
                routing_threshold=0.35,

                instructions=(
                    load_prompt("tools/file_inspection")
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

                routing_profile=load_prompt("routing/toolsets/file_editing"),
                routing_threshold=0.35,

                instructions=(
                    load_prompt("tools/file_editing")
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

                routing_profile=load_prompt("routing/toolsets/software_development"),
                routing_threshold=0.35,

                instructions=(
                    load_prompt("tools/software_development")
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
            ToolsetSpec(
                name="APPWORLD",
                description="在隔离的 AppWorld 模拟应用环境中查阅 API、执行 Python 并完成跨应用任务；仅用于评测环境。",
                routing_profile=load_prompt("routing/toolsets/appworld"),
                routing_threshold=0.35,
                instructions="使用 appworld_discover查API目录和签名，再用appworld_execute操作模拟应用；不得把模拟数据当成真实账户数据，也不得声称已操作外部真实服务。",
                # General/Code Worker看到execute，Code Reviewer看到verify。
                # discover是三者共同的入口；其余两个按角色实际存在时展开。
                required_tool_names=("appworld_discover",),
                optional_tool_names=("appworld_execute", "appworld_verify"),
                exclusive=True,
            ),
        ]
    )
)
