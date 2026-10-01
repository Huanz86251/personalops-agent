from __future__ import annotations

from prompt_loader import load_prompt
from amap_mcp_runtime import (
    AMAP_ACTION_LINK_TOOL_NAMES,
    AMAP_LOCATION_CONTEXT_TOOL_NAMES,
    AMAP_POI_TOOL_NAMES,
    AMAP_ROUTE_TOOL_NAMES,
    AMAP_TOOL_NAMES,
)

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

    # False 表示该能力只能在 Worker 执行前的首次路由中选中，
    # 不得被执行中的 request_additional_capabilities 当作回退路径。
    allow_runtime_expansion: bool = True

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

        required_resolutions = {
            name: name if name in tools_by_name else None
            for name in spec.required_tool_names
        }
        optional_resolutions = {
            name: name if name in tools_by_name else None
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
                name="HOST_FILE_INTERACTION",
                description=(
                    "按用户要求浏览或搜索电脑上的既有文件，并在独立审批后将精确文件"
                    "只读导入当前Worker隔离工作区；不等同于向外发送文件。"
                ),
                routing_profile=load_prompt("routing/toolsets/host_file_interaction"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/host_file_interaction"),
                required_tool_names=(
                    "host_list_files", "host_search_files", "host_import_file",
                ),
            ),
            ToolsetSpec(
                name="FEISHU_FILE_EXPORT",
                description="授权用户通过任一执行Worker将本机文件、下载候选、已审核handoff或工作区成品发回当前飞书会话；每次外发由独立Judge核对原始请求，宿主执行身份、完整性和大小校验（受飞书平台30 MB上限约束）。",
                routing_profile=load_prompt("routing/toolsets/feishu_file_export"),
                # “在飞书提醒我”也包含“飞书”，但不是文件回传。
                # Secondary候选还有0.05共享容差；实际文件回传样例约0.56，
                # 提醒样例约0.45，因此用0.52阻断后者并保留前者。
                routing_threshold=0.52,
                instructions="使用 send_local_file_to_feishu 提交本机绝对路径、/downloads候选、/handoff文件或当前工作区路径和具体发送理由；Harness对每次外发独立审批，存疑时暂停等待用户允许或拒绝。只有工具返回 SENT 和 message_id 才算交付。USER_DENIED不可重试或绕过。工具报大小超限时停止，向用户说明未上传并询问是否允许拆分。",
                required_tool_names=("send_local_file_to_feishu",),
            ),
            ToolsetSpec(
                name="DOCUMENT_READING",
                description=(
                    "读取和提取本地PDF、Word、PPTX、任务附件与扫描图片中的文字；"
                    "支持分页、OCR和带页码的材料核对，但不修改表格或转换文件格式。"
                ),
                routing_profile=load_prompt("routing/toolsets/document_reading"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/document_reading"),
                required_tool_names=("attachment_to_text", "ocr_image"),
            ),
            ToolsetSpec(
                name="DOCUMENT_CONVERSION",
                description=(
                    "把本地文档在PDF、DOCX、PPTX等受支持格式之间转换并生成新文件；"
                    "用于明确的格式转换或导出，不负责阅读总结、电子表格编辑或发送文件。"
                ),
                routing_profile=load_prompt("routing/toolsets/document_conversion"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/document_conversion"),
                required_tool_names=("convert_document",),
            ),
            ToolsetSpec(
                name="SPREADSHEET_WORK",
                description=(
                    "读取和修改Excel电子表格的单元格、公式、样式与图表；"
                    "适用于XLSX表格分析和结构化编辑，不用于普通文本、PDF阅读或网页表格搜索。"
                ),
                routing_profile=load_prompt("routing/toolsets/spreadsheet_work"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/spreadsheet_work"),
                required_tool_names=(
                    "spreadsheet_read", "spreadsheet_write",
                    "spreadsheet_format", "spreadsheet_chart",
                ),
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
                name="SCHEDULE_CREATION",
                description=(
                    "创建一次、每日或每周的持久提醒，或立即发送Windows通知；"
                    "支持到期Windows通知、当前飞书会话文本和一次性Agent队列任务，"
                    "但不创建外部日历会议，也不定时发送邮件。"
                ),
                routing_profile=load_prompt("routing/toolsets/schedule_creation"),
                # Secondary candidates receive a small shared tolerance in
                # ToolsetRouter. Keep this group at 0.45 so unrelated
                # multi-tool tasks scoring around 0.36 cannot slip in, while
                # representative reminder intents remain above 0.56.
                routing_threshold=0.45,
                instructions=load_prompt("tools/schedule_creation"),
                required_tool_names=(
                    "get_current_time",
                    "schedule_create",
                    "schedule_create_feishu_reminder",
                    "schedule_create_agent_task",
                    "windows_notify",
                ),
            ),
            ToolsetSpec(
                name="SCHEDULE_MANAGEMENT",
                description=(
                    "查询已有持久提醒及运行记录，并按真实schedule_id暂停、恢复或取消；"
                    "用于管理已经存在的提醒，不负责创建日历会议、发送邮件或执行任意后台脚本。"
                ),
                routing_profile=load_prompt("routing/toolsets/schedule_management"),
                routing_threshold=0.45,
                instructions=load_prompt("tools/schedule_management"),
                required_tool_names=(
                    "schedule_list",
                    "schedule_pause",
                    "schedule_resume",
                    "schedule_delete",
                    "schedule_runs",
                ),
            ),
            ToolsetSpec(
                name="EMAIL_READING",
                description=(
                    "通过本地只读IMAP连接查看QQ、Foxmail或其他兼容邮箱的"
                    "连接状态、近期邮件、正文摘要和附件，并可将用户指定附件保存到"
                    "本机隔离目录；不能写草稿、发送、删除、移动或标记邮件。"
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
                ),
            ),
            ToolsetSpec(
                name="EMAIL_DRAFTING",
                description=(
                    "在用户明确要求后，把已确认的收件人、主题和纯文本正文保存到"
                    "邮箱Drafts草稿箱；只保存草稿，绝不发送邮件，也不读取收件箱或下载附件。"
                ),
                routing_profile=load_prompt("routing/toolsets/email_drafting"),
                routing_threshold=0.40,
                instructions=load_prompt("tools/email_drafting"),
                required_tool_names=(
                    "email_create_draft",
                ),
            ),
            ToolsetSpec(
                name="ECOMMERCE_1688",
                description=(
                    "在1688寻找商品和货源，查询带商品链接、起订量和阶梯价格的"
                    "B2B报价；适合选品和批发价格参考，不代表淘宝零售价。"
                ),
                routing_profile=load_prompt("routing/toolsets/ecommerce_1688"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/ecommerce_1688"),
                required_tool_names=("search_1688_products",),
                exclusive=False,
            ),
            ToolsetSpec(
                name="AMAP_POI",
                description=(
                    "高德地点与兴趣点搜索：按城市、区县、地址、关键词或坐标查找餐馆、"
                    "健身房、医院、商场、景点等实体，查询周边地点及POI详情。"
                    "用于本地服务候选和地点核验，不证明评分、口碑或实时营业状态。"
                ),
                routing_profile=load_prompt("routing/toolsets/amap_poi"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/amap_poi"),
                required_tool_names=AMAP_POI_TOOL_NAMES,
                exclusive=False,
            ),
            ToolsetSpec(
                name="AMAP_ROUTE",
                description=(
                    "高德路线与距离：把用户给出的起点和终点解析为地点，查询驾车、公交、"
                    "步行或骑行路线、里程与预计耗时；不用于餐馆口碑、天气或真实叫车下单。"
                ),
                routing_profile=load_prompt("routing/toolsets/amap_route"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/amap_route"),
                required_tool_names=AMAP_ROUTE_TOOL_NAMES,
                exclusive=False,
            ),
            ToolsetSpec(
                name="AMAP_LOCATION_CONTEXT",
                description=(
                    "高德地点上下文：地址与坐标互转、粗略IP定位以及指定城市的天气查询；"
                    "用于地理编码、逆地理编码、地区识别和天气，不搜索本地商户或规划路线。"
                ),
                routing_profile=load_prompt("routing/toolsets/amap_location_context"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/amap_location_context"),
                required_tool_names=AMAP_LOCATION_CONTEXT_TOOL_NAMES,
                exclusive=False,
            ),
            ToolsetSpec(
                name="AMAP_ACTION_LINKS",
                description=(
                    "生成可交给高德地图客户端的个人地图、导航或打车链接；"
                    "只在用户明确要求生成地图/导航/打车入口时选择，生成链接不代表已经导航、叫车或下单。"
                ),
                routing_profile=load_prompt("routing/toolsets/amap_action_links"),
                routing_threshold=0.45,
                instructions=load_prompt("tools/amap_action_links"),
                required_tool_names=AMAP_ACTION_LINK_TOOL_NAMES,
                exclusive=False,
            ),
            ToolsetSpec(
                name="MOVIE",
                description=(
                    "电影专用：查询城市影院、指定日期的影片、"
                    "各影院场次与逐场票价。"
                ),
                routing_profile=load_prompt("routing/toolsets/movie"),
                routing_threshold=0.35,
                instructions=load_prompt("tools/movie"),
                required_tool_names=(
                    "get_current_time",
                    "list_cinemas",
                    "list_city_movies",
                    "get_cinema_showtimes",
                    "get_movie_showtimes",
                ),
                optional_tool_names=(
                    "web_search",
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
                ),
            ),

            ToolsetSpec(
                name="BROWSER_AUTOMATION",

                description=(
                    "高成本真实浏览器控制：仅当用户或当前Step明确要求"
                    "登录、点击、输入、选择、提交或多页面交互时使用；"
                    "不用于普通搜索、资料查询、推荐、静态阅读或搜索失败后的回退。"
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
                allow_runtime_expansion=False,
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
                    "ls",
                    "glob",
                    "grep",
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
                    "grep",
                    "edit_file",
                    "write_file",
                    "glob",
                ),

                optional_tool_names=(
                    "ls",
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
                    "grep",
                    "execute",
                ),

                optional_tool_names=(
                    "glob",
                    "ls",
                    "edit_file",
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
