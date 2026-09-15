"""Task-independent contracts shared by AppWorld discovery/execution tools."""

ENTRY = """Before submitting, check reason against code. If the cited API or argument source is missing, call appworld_discover first; do not emit draft calls or guessed business code. A reason is a short evidence statement, not a reasoning transcript.
Python runs in the existing persistent task world; the `apis` object is already available.
Use interfaces supported by a visible user/RAG/runtime source, a successful appworld_discover result, or a prior successful AppWorld call. A citation proves provenance, not correctness: check the cited scope and exact signature before acting.
Never guess, compose or autocomplete API/method names. If the exact name or signature is unconfirmed, discover it first; if unavailable, report the missing information instead of trying an invented name.
Look up only missing prerequisites; once available, execute the next task action.
Call documented APIs as apis.<app_name>.<api_name>(keyword=value); print needed results.
No import, new AppWorld instance, filesystem exploration or network setup is needed.
Use supervisor's documented APIs for simulated account information. Argument values must come from user-provided facts, observed results or documented defaults; examples and placeholders are not task data. Resolve missing required values or report the blocker instead of guessing.
Docker files and variables are separate. Never access hidden evaluator data."""

DISCOVER_DESCRIPTION = """Discover exact AppWorld API names and signatures before business execution.
This tool only accepts Python that calls apis.api_docs.* documentation methods and prints their results.
It cannot query or modify business application data. Use the returned evidence reference in a later
appworld_execute call. Do not guess an API name and do not put business API calls in this tool."""

EXECUTE_DESCRIPTION = "Execute Python to query or modify the current simulated AppWorld. For a Step with target_selection, classify action_phase; TARGET_READ binds every target set to its exact read API, while TARGET_WRITE and TARGET_VERIFY cite the frozen read receipt in binding_ref. " + ENTRY
VERIFY_DESCRIPTION = ("Independently verify this same AppWorld for Code Reviewer. " + ENTRY +
    "\nUse explicit inputs and documented queries; do not repeat business writes or repair data. "
    "Login/documentation calls may be needed. This tool is not an enforced read-only sandbox; request Worker repair for defects.")


from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppWorldDiscoverInput(BaseModel):
    source_refs: list[str] = Field(
        min_length=1,
        max_length=4,
        description=(
            "为什么需要本次查找：复制当前可见的短来源号，例如U1、R1、E2；"
            "若只是根据通用知识决定先查文档，可填MODEL。不要编造编号。"
        ),
    )
    reason: str = Field(
        min_length=1,
        description="一句话说明尚缺哪个应用、API名称或签名；不要写思维过程。",
    )
    code: str = Field(
        description="只调用apis.api_docs.*并打印结果的Python；不得调用业务应用API。",
    )

AppWorldActionPhase = Literal[
    "PREREQUISITE",
    "TARGET_READ",
    "TARGET_WRITE",
    "TARGET_VERIFY",
    "FINALIZE",
    "OTHER",
]


class AppWorldCoveragePlan(BaseModel):
    reason: str = Field(
        min_length=1,
        max_length=300,
        description="先说明为什么这个停止条件足以证明目标范围已经读取完整。",
    )
    completion_condition: str = Field(
        min_length=1,
        max_length=300,
        description="读取前定义何时算完整，例如读到无下一页、返回明确总数并对齐，或接口保证一次返回全部。",
    )


class AppWorldSetBinding(BaseModel):
    set_id: str = Field(
        min_length=1,
        max_length=24,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="复制当前Step.target_selection中的集合编号，例如A或B。",
    )
    read_api: str = Field(
        min_length=3,
        max_length=160,
        pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$",
        description=(
            "最初产生该集合的真实读取接口，格式app.api，必须来自已见文档。"
            "本对象只在TARGET_READ中填写；后续阶段通过binding_ref引用冻结回执，"
            "不要改填写入接口或回读接口，也不能猜。"
        ),
    )
    source_ref: str = Field(
        pattern=r"^(?:P1|[URSHCE][1-9][0-9]*)$",
        description="支持本次TARGET_READ绑定的可见来源号，引用已见接口文档或真实返回。",
    )
    match_reason: str = Field(
        min_length=1,
        max_length=300,
        description="一句话比较接口真实返回范围与集合definition，说明没有额外加入或丢失筛选条件。",
    )
    requested_scope: str = Field(
        min_length=1,
        max_length=400,
        description="读取前写清楚本次要取得的对象边界、筛选条件和是否要求全部结果。",
    )
    coverage_plan: AppWorldCoveragePlan = Field(
        description="与requested_scope放在一起，读取前定义怎样证明结果完整。",
    )


class AppWorldBindingCheck(BaseModel):
    reason: str = Field(
        min_length=1,
        max_length=300,
        description="先比较当前集合definition与上一轮显示的真实接口描述，说明两者为何相同或不同；不要只复述接口名。",
    )
    assessment: Literal[
        "COMPLETE_MATCH",
        "INCOMPLETE_RESULT",
        "WRONG_READ",
        "CONTRACT_CONFLICT",
    ] = Field(
        description=(
            "完成reason后再判断。COMPLETE_MATCH才允许写入；INCOMPLETE_RESULT表示范围方向正确但未读完整；"
            "WRONG_READ表示读取对象或筛选范围错误；CONTRACT_CONFLICT表示上游集合定义疑似违背用户请求。"
        ),
    )
    set_id: str = Field(
        min_length=1,
        max_length=24,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="本判断对应的target_selection集合编号。",
    )


class AppWorldCallInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "source_refs": ["U1"],
                "reason": "先取得登录等准备条件。",
                "action_phase": "PREREQUISITE",
                "binding_ref": None,
                "binding_checks": [],
                "set_bindings": [],
                "code": "# 准备动作代码；无集合绑定也必须显式填写[]",
            },
            {
                "source_refs": ["E1"],
                "reason": "接口来自E1文档；范围来自当前Step。",
                "action_phase": "TARGET_READ",
                "binding_ref": None,
                "binding_checks": [],
                "set_bindings": [
                    {"set_id": "A", "read_api": "drive.list_folder_documents",
                     "source_ref": "E1", "match_reason": "产生文件夹内文档集合，没有加入批准条件。",
                     "requested_scope": "选定文件夹内的全部文档",
                     "coverage_plan": {"reason": "接口分页，因此读到无下一页才不会漏项。", "completion_condition": "持续分页直到没有下一页"}},
                    {"set_id": "B", "read_api": "drive.list_approved_documents",
                     "source_ref": "E1", "match_reason": "产生已批准文档集合，没有加入文件夹条件。",
                     "requested_scope": "全部已批准文档",
                     "coverage_plan": {"reason": "接口保证一次返回全部匹配项。", "completion_condition": "一次返回且响应没有分页字段"}},
                ],
                "code": "# 示例API仅展示结构；实际调用必须替换为已见文档中的真实API",
            },
            {
                "source_refs": ["E5"],
                "reason": "只写入E5冻结的A与B交集。",
                "action_phase": "TARGET_WRITE",
                "binding_ref": "E5",
                "binding_checks": [
                    {"reason": "A的真实描述返回文件夹内文档，与A的definition一致。", "assessment": "COMPLETE_MATCH", "set_id": "A"},
                    {"reason": "B的真实描述返回已批准文档，与B的definition一致。", "assessment": "COMPLETE_MATCH", "set_id": "B"},
                ],
                "set_bindings": [],
                "code": "# Harness从E5恢复冻结集合；这里只写对该集合的操作",
            },
            {
                "source_refs": ["E5"],
                "reason": "回读E5冻结的同一交集。",
                "action_phase": "TARGET_VERIFY",
                "binding_ref": "E5",
                "binding_checks": [],
                "set_bindings": [],
                "code": "# Harness从E5恢复冻结集合；回读并逐项核对同一集合",
            },
            {
                "source_refs": ["E8"], "reason": "E8证明完整回读成功。",
                "action_phase": "FINALIZE", "binding_ref": None, "binding_checks": [], "set_bindings": [],
                "code": "print(apis.supervisor.complete_task())",
            },
        ],
    })
    source_refs: list[str] = Field(
        min_length=1,
        max_length=4,
        description=(
            "复制支持本次调用的短来源号，可来自用户、RAG、运行状态、Discover或此前真实返回。"
            "引用只说明来源，不保证内容正确；执行前仍须核对范围和签名。不要编造编号，"
            "MODEL不能单独证明环境API名称、签名或业务参数。"
        ),
    )
    reason: str = Field(min_length=1, description="先用一句话说明两项依据：API名称和签名来自哪个已见文档/Skill/工具返回；参数值来自哪个真实返回或用户输入。可引用当前可见的E编号，不编造引用。仅查文档时说明待补哪项依据，无业务参数可写不涉及。")
    action_phase: AppWorldActionPhase = Field(
        description=(
            "本次动作阶段，必须显式填写。登录、凭据等准备动作选PREREQUISITE；"
            "第一次构造目标集合选TARGET_READ；对冻结目标写入选TARGET_WRITE；"
            "写后回读选TARGET_VERIFY；最后单独complete_task选FINALIZE。"
            "不要把登录、目标读取、写入、回读或结案混成同一阶段。"
        ),
    )
    binding_ref: str | None = Field(
        default=None,
        pattern=r"^E[1-9][0-9]*$",
        description=(
            "TARGET_WRITE和TARGET_VERIFY必须填写同一次成功TARGET_READ返回的E编号；"
            "Harness会从该回执恢复并校验冻结的集合绑定。TARGET_READ、PREREQUISITE、"
            "FINALIZE和普通任务填null。必须同时把该编号放进source_refs。"
        ),
    )
    @field_validator("binding_ref", mode="before")
    @classmethod
    def normalize_null_binding_ref(cls, value):
        # Providers sometimes serialize an empty optional value as a string or
        # boolean sentinel.  Treat only conventional null-like values as absent;
        # malformed evidence IDs must still fail loudly.
        if value is False:
            return None
        if isinstance(value, str) and value.strip().lower() in {
            "", "null", "none", "nil", "false", "n/a",
        }:
            return None
        return value

    binding_checks: list[AppWorldBindingCheck] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "必须显式填写。仅MUTATION的TARGET_WRITE逐集合填写：先写reason比较上一轮动态显示的真实描述与集合definition，"
            "再填assessment，最后填set_id。只有全部COMPLETE_MATCH才执行写入；INCOMPLETE_RESULT或WRONG_READ会让Harness"
            "不执行代码、废弃旧读取回执并要求同一Worker重新TARGET_READ；CONTRACT_CONFLICT会阻止写入并要求报告上游范围冲突。"
            "其他阶段及READ_ONLY填[]。"
        ),
    )
    set_bindings: list[AppWorldSetBinding] = Field(
        max_length=4,
        description=(
            "必须显式填写。当前Step有target_selection时：TARGET_READ为operands中每个集合各填一次；"
            "TARGET_WRITE和TARGET_VERIFY填[]，只通过binding_ref引用冻结的TARGET_READ回执；"
            "Harness负责恢复绑定。PREREQUISITE、FINALIZE和普通任务也填[]。"
            "本字段描述集合血缘，不描述当前代码调用的写入/回读API。参照schema examples的四阶段结构，"
            "但绝对不要复制示例API名。"
        ),
    )
    code: str = Field(description="上述依据对应的Python代码；缺少接口或必填参数依据时只补查，不猜测执行。")
