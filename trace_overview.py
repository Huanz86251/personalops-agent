"""Read-only trace presentation. Never feeds summaries back into an agent."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from threading import RLock
from collections.abc import Mapping


def decoded(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    return value


def tool_purpose(name, arguments):
    """Deterministic labels from the actual tool and arguments; not a model claim."""
    args = decoded(arguments)
    code = str((args or {}).get("code", "")) if isinstance(args, dict) else ""
    if name in {"appworld_discover", "appworld_execute", "appworld_verify"}:
        if "api_docs." in code:
            return "AppWorld / API 文档"
        return "AppWorld / 独立核验" if name == "appworld_verify" else "AppWorld / 执行业务代码"
    return {
        "execute": "Docker / 执行代码与测试",
        "submit_code_for_review": "Code Worker / 提交候选",
        "publish_reviewed_candidate": "Code Reviewer / 发布交付物",
        "submit_code_review": "Code Reviewer / 提交审核结论",
        "report_general_result": "General Agent / 提交结果",
    }.get(name, "Tool / " + name)


ROLE_LABELS = {
    "scheduler": "Scheduler · 规划", "general": "General Agent",
    "code": "Code Worker", "code_reviewer": "Code Reviewer",
    "reviewer": "Code Reviewer", "final_reviewer": "Scheduler · 最终验收",
    "web": "Web Agent", "reporter": "General Reviewer",
    "replanner": "Scheduler · 重规划",
    "scope_resolver": "Scope Resolver", "skill_selector": "Skill Selector",
}


def build_overview(task, result, rows, *, preview=False, capture_limited=False):
    """Return human overview plus every original model input/output at the root."""
    calls = defaultdict(list)
    plans, reports, skills = [], [], []
    tools = []
    for row in rows:
        attrs = row.get("attributes", {})
        kind = attrs.get("openinference.span.kind")
        incoming, outgoing = decoded(attrs.get("input.value")), decoded(attrs.get("output.value"))
        if kind == "LLM":
            role = attrs.get("runtime.model_role", "unknown")
            calls[role].append({
                "调用序号": len(calls[role]) + 1, "节点": row["name"],
                "输入原文": incoming, "输出原文": outgoing,
                "耗时毫秒": attrs.get("timing.duration_ms"),
                "首内容毫秒": attrs.get("timing.first_token_ms"),
                "usage": outgoing.get("usage", {}) if isinstance(outgoing, dict) else {},
            })
            responses = outgoing.get("responses", []) if isinstance(outgoing, dict) else []
            for response in responses:
                value = decoded(response.get("content"))
                if isinstance(value, dict) and value.get("steps"):
                    plans.append(value)
                for call in response.get("tool_calls", []):
                    args = call.get("args", {})
                    report = args.get("report") or args.get("result") or args.get("submission")
                    if isinstance(report, dict):
                        reports.append({"role": role, "tool": call.get("name"), "report": report})
        elif kind == "TOOL":
            tools.append({"name": attrs.get("tool.name", row["name"]), "status": attrs.get("business.status", "UNKNOWN")})
        if row["name"].startswith("Skills /"):
            skills.append({"节点": row["name"], "输入": incoming, "输出": outgoing})

    grade = result.get("official_evaluation")
    grade_text = "未评分" if grade is None else ("通过" if grade.get("success") else "未通过")
    mode = "历史记录回放预览 · 没有新任务执行 · 不计入准确率" if preview else "任务执行记录 · 官方评分与模型自评分开"
    if isinstance(grade, dict) and grade.get("num_tests") is not None:
        grade_text += f"（{len(grade.get('passes', []))}/{grade['num_tests']}）"
    md = ["# 任务总览", f"**{mode}**", "", f"**Scheduler 结论：{result.get('scheduler_status', '未知')} ｜ 官方评分：{grade_text}**", "",
          "## 用户原始任务", str(task.get("instruction", "")), "",
          "## 最终结果", str(result.get("answer") or result.get("error") or "尚未完成"), "",
          "## 计划路线（实际执行看下表与节点）", "```text", "用户任务", "└─ Scheduler 规划"]
    for plan in plans:
        for step in plan.get("steps", []):
            role = step.get("worker_kind", "UNKNOWN")
            md.append(f"   ├─ Step {step.get('step_id')} · {role}")
            if role == "CODE":
                md.extend(["   │  ├─ Code Worker → 编写 / 自测 / 执行业务 → 提交候选",
                           "   │  └─ Code Reviewer → 独立核验 → 发布 → 审核报告"])
    md.extend(["   └─ Scheduler Final Review → 锁定任务 → AppWorld Judge", "```", "",
               "## 各角色调用与用量", "| 角色 | 模型调用 | 输入 token | 输出 token | 缓存 token | 推理 token |",
               "| --- | ---: | ---: | ---: | ---: | ---: |"])
    def usage_sum(items, key):
        values = [item["usage"].get(key) for item in items]
        known = [v for v in values if isinstance(v, int)]
        if len(known) != len(values):
            return "未知" if not known else f"已知 {sum(known)} / 缺 {len(values)-len(known)} 次"
        return str(sum(known))
    for role, items in calls.items():
        md.append("| " + " | ".join([ROLE_LABELS.get(role, role), str(len(items)),
            *(usage_sum(items, key) for key in ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens"))]) + " |")
    from cost_accounting import summarize_cost
    md.extend(["", "## 费用估算（人民币）", "| 角色 | 费用（元） | 未计价调用 |", "| --- | ---: | ---: |"])
    all_costs = []
    for role, items in calls.items():
        costs = [item["usage"].get("cost", {}) for item in items]
        all_costs.extend(costs)
        subtotal = summarize_cost(costs)
        amount = subtotal["known_total_cny"]
        amount_text = "未知" if amount is None else f"¥{amount:.8f}" + ("（部分）" if subtotal["missing_requests"] else "")
        md.append(f"| {ROLE_LABELS.get(role,role)} | {amount_text} | {subtotal['missing_requests']} |")
    total_cost = summarize_cost(all_costs)
    amount = total_cost["known_total_cny"]
    md.append("总计：" + ("未知" if amount is None else f"¥{amount:.8f}") + f"；未计价 {total_cost['missing_requests']} 次。")
    md.append("按公开单价估算，非实际账单。Phoenix 顶部 Total Cost 为美元换算；单价和汇率日期见每次调用 Raw 的 usage.cost。")
    md.extend(["", "缓存属于输入、推理属于输出，不能重复相加。未知不是 0；未配置单价不推算费用。",
               "回放/人工替身的耗时不代表真实模型速度，回放不产生模型费用。" if preview else "节点耗时和首内容时间见各调用；没有流式数据时首内容时间未知。",
               "", "## 角色范围耗时", "| 范围 | 秒 |", "| --- | ---: |"])
    for row in rows:
        name = row["name"]
        if any(label in name for label in ("General Agent / Step", "Code Worker / Step", "Code Reviewer / Step", "Code Agent / Step")) or name in {"🎛️ Scheduler", "Scheduler / Plan", "Scheduler / Final Review", "AppWorld Judge"}:
            elapsed = row.get("attributes", {}).get("timing.duration_ms")
            md.append(f"| {name} | {elapsed/1000:.3f} |" if isinstance(elapsed, (int,float)) else f"| {name} | 未知 |")
    md.extend(["", "父范围包含子范围，耗时不能相加。回放仅展示历史录制时长。" if preview else "父范围包含子范围，耗时不能相加。", "", "## 审核与交付依据"])
    for item in reports:
        report = item["report"]
        md.extend([f"### {ROLE_LABELS.get(item['role'], item['role'])} · {item['tool']}",
                   str(report.get("summary", ""))])
        for check in report.get("check_results", []):
            md.append(f"- {check.get('status', '未知')} · {check.get('description', '')}：{check.get('summary', '')}")
        if report.get("publication_id"):
            md.append("发布回执：" + str(report["publication_id"]))
    md.extend(["", "## 工具、Skills 与记录边界",
               f"工具调用 {len(tools)} 次；Skills 准备节点 {len(skills)} 个（准备节点不等于加载了 Skill）。",
               f"汇集了 {len(rows)} 个节点记录、{sum(map(len, calls.values()))} 次模型调用。",
               "完整输入与输出集中在本根节点的 Input：按角色、调用序号展开，无须逐个点击树节点。",
               "本节点 Input 同时保留原始任务、计划原文、审核报告及 Skills 记录；树节点保留工具原始证据。",
               "捕获达到保护上限，完整记录以子节点为准。" if capture_limited else "没有因展示而删减捕获到的模型输入/输出。"])
    for skill in skills:
        value = skill["输出"]
        if isinstance(value, dict):
            md.append(f"- {value.get('role', skill['节点'])}：mode={value.get('mode', '未知')}；selected={value.get('selected', '未知')}")
    root_input = {"01 用户原始任务": task.get("instruction", ""),
                  "02 计划原文": plans, "03 按角色查看每次完整上下文与输出": dict(calls),
                  "04 审核与交付报告": reports, "05 Skills 加载记录": skills,
                  "06 展示模式": mode, "07 捕获是否达到上限": capture_limited}
    return "\n".join(md), root_input


def _plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _brief(value, limit=130):
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _conversation_trace_tree(rows, *, max_nodes=65):
    """Show real semantic ancestry, never a fabricated sequential plan."""
    by_id = {row.get("span_id"): row for row in rows if row.get("span_id")}
    selected = {}
    for span_id, row in by_id.items():
        name = str(row.get("name") or "")
        if (
            name == "🎛️ Scheduler"
            or name in {"Scheduler / Plan", "Scheduler / Dispatch Step",
                        "Scheduler / Replan", "Scheduler / Final Review",
                        "Scheduler / Accept Code Review"}
            or name.startswith(("Scheduler / Review", "Step Reporter /"))
            or ("/ Step " in name and any(role in name for role in
                ("General Agent", "Web Agent", "Code Agent", "Code Worker", "Code Reviewer")))
        ):
            selected[span_id] = row

    children = defaultdict(list)
    for span_id, row in selected.items():
        parent_id = row.get("parent_id")
        seen = {span_id}
        while parent_id in by_id and parent_id not in selected and parent_id not in seen:
            seen.add(parent_id)
            parent_id = by_id[parent_id].get("parent_id")
        children[parent_id if parent_id in selected else None].append(span_id)
    for siblings in children.values():
        siblings.sort(key=lambda span_id: str(by_id[span_id].get("start_time") or ""))

    def counts(root_id):
        llm = tools = 0
        for row in rows:
            parent_id = row.get("parent_id")
            seen = set()
            while parent_id in by_id and parent_id not in seen:
                if parent_id == root_id:
                    kind = row.get("attributes", {}).get("openinference.span.kind")
                    llm += kind == "LLM"
                    tools += kind == "TOOL"
                    break
                seen.add(parent_id)
                parent_id = by_id[parent_id].get("parent_id")
        return llm, tools

    lines = []
    def render(span_id, prefix="", is_last=True):
        if len(lines) >= max_nodes:
            return
        row = selected[span_id]
        name = row["name"]
        attrs = row.get("attributes", {})
        output = decoded(attrs.get("output.value"))
        if not isinstance(output, dict):
            output = attrs.get("trace_overview.step")
        if name == "Scheduler / Dispatch Step" and isinstance(output, dict):
            step = _plain(output.get("current_step"))
            if step:
                name += f" {step.get('step_id', '?')} · {step.get('worker_kind', '?')}"
            if output.get("current_step_attempt"):
                name += f" · 第 {output['current_step_attempt']} 次尝试"
        llm, tools = counts(span_id)
        elapsed = attrs.get("timing.duration_ms")
        duration = ""
        if isinstance(elapsed, (int, float)):
            duration = (f" · {elapsed:.0f}ms" if elapsed < 1000
                        else f" · {elapsed / 1000:.1f}s")
        usage = f" · 模型 {llm} / 工具 {tools}" if llm or tools else ""
        lines.append(prefix + ("└─ " if is_last else "├─ ") + name + duration + usage)
        descendants = children.get(span_id, [])
        next_prefix = prefix + ("   " if is_last else "│  ")
        for index, child_id in enumerate(descendants):
            render(child_id, next_prefix, index == len(descendants) - 1)

    roots = children.get(None, [])
    for index, span_id in enumerate(roots):
        render(span_id, "", index == len(roots) - 1)
    if len(selected) > len(lines):
        lines.append(f"…另外 {len(selected) - len(lines)} 个流程节点请在原始调用树展开。")
    return lines or ["（本轮没有捕获到规划流程节点；原始子节点仍保留。）"]


def build_conversation_overview(task, result, rows, *, capture_limited=False):
    """A bounded, presentation-only index for a normal conversation turn."""
    planning = _plain(result.get("planning"))
    steps = [_plain(item) for item in planning.get("planned_steps", [])]
    reports = [_plain(item) for item in planning.get("completed_step_reports", [])]
    latest_report = {str(item.get("step_id")): item for item in reports}
    llm_rows = [row for row in rows if row.get("attributes", {}).get("openinference.span.kind") == "LLM"]
    tool_rows = [row for row in rows if row.get("attributes", {}).get("openinference.span.kind") == "TOOL"]
    status = str(planning.get("final_status") or result.get("status") or "UNKNOWN")
    md = ["# 对话任务总览", "**这是 Harness 根据实际 Trace 整理的只读索引；原始输入、工具回执和错误仍在子节点。**", "",
          f"**终态：{status} · 模型调用 {len(llm_rows)} 次 · 工具调用 {len(tool_rows)} 次**", "",
          "## 本轮用户要求", str(task.get("instruction") or ""), "", "## 发给用户的回答",
          str(result.get("answer") or result.get("error") or "（本轮没有最终回答）"), "",
          "## 计划与 Step 验收"]
    if steps:
        for step in steps:
            report = latest_report.get(str(step.get("step_id")), {})
            md.append(f"- Step {step.get('step_id', '?')} · {step.get('worker_kind', '?')} · "
                      f"{report.get('status') or '未形成 StepReport'}：{_brief(step.get('objective'), 200)}")
            if report.get("summary"):
                md.append(f"  - 验收：{_brief(report['summary'], 240)}")
    else:
        md.append("- 本轮没有生成需要执行的 Step；可能由 Scheduler 直接答复或在规划前终止。")
    by_role = defaultdict(list)
    for row in llm_rows:
        attrs = row.get("attributes", {})
        by_role[str(attrs.get("runtime.model_role") or "unknown")].append(
            attrs.get("trace_overview.usage") or {}
        )
    if by_role:
        md.extend(["", "## 模型用量（缓存读取属于输入，不重复相加）",
                   "| 角色 | 调用 | 输入 token | 输出 token | 缓存读取 token | 估算费用（元） |",
                   "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for role, usages in by_role.items():
            def measured(key):
                known = [item[key] for item in usages if isinstance(item.get(key), (int, float))]
                return (str(int(sum(known))) if len(known) == len(usages)
                        else (f"已知 {int(sum(known))} / 缺 {len(usages)-len(known)} 次"
                              if known else "未知"))
            costs = [item["cost_cny"] for item in usages
                     if isinstance(item.get("cost_cny"), (int, float))]
            cost_text = (f"¥{sum(costs):.4f}" if len(costs) == len(usages)
                         else (f"已知 ¥{sum(costs):.4f} / 缺 {len(usages)-len(costs)} 次"
                               if costs else "未知"))
            md.append(f"| {ROLE_LABELS.get(role, role)} | {len(usages)} | "
                      f"{measured('input_tokens')} | {measured('output_tokens')} | "
                      f"{measured('cache_read_tokens')} | {cost_text} |")
        md.append("费用按记录中的公开单价估算，不是实际账单；缺失用量不当作零。")
    md.extend(["", "## 实际调用树（同层分支可并行；时间为范围耗时，不能相加）", "```text",
               *_conversation_trace_tree(rows), "```", "",
               "## 阅读边界", "计划是 Scheduler 的预期，StepReport 是验收记录；两者都不能替代原始工具证据。",
               "并行 Worker 按真实父子 Span 归组，不按结束顺序伪装成串行。"])
    if capture_limited:
        md.append("本总览达到 1000 节点保护上限；请到原始 Trace 查看剩余节点。")
    index = [{"span_id": row.get("span_id"), "name": row.get("name"),
              "role": row.get("attributes", {}).get("runtime.model_role")}
             for row in llm_rows]
    root_input = {"01 用户原始要求": task.get("instruction"), "02 实际计划": steps,
                  "03 Step 验收报告": reports, "04 模型调用索引": index,
                  "05 原始证据位置": "展开此 Trace 的子节点；原顶层结果见 Conversation / Result Details；总览不复制多轮完整上下文。"}
    return "\n".join(md), root_input


@contextmanager
def conversation_overview(span, task, result, *, enabled=True):
    """Attach a readable root to non-AppWorld turns without changing execution."""
    if not enabled:
        yield
        return
    from observability import set_span_input, set_span_output, trace_span
    with capture_task(span, compact=True) as captured:
        try:
            yield
        except BaseException as error:
            result.setdefault("error", str(error))
            result.setdefault("status", "INTERRUPTED")
            raise
        finally:
            try:
                original_input = getattr(span, "attributes", {}).get("input.value")
                original_output = getattr(span, "attributes", {}).get("output.value")
                if original_output is not None:
                    with trace_span("Conversation / Result Details") as details:
                        if original_input is not None:
                            set_span_input(details, decoded(original_input))
                        set_span_output(details, decoded(original_output))
                summary, inputs = build_conversation_overview(
                    task, result, captured["rows"], capture_limited=captured["limited"]
                )
                set_span_input(span, inputs)
                set_span_output(span, summary)
            except Exception:
                import logging
                logging.getLogger("agent").exception(
                    "Conversation trace overview unavailable; original spans remain"
                )


_LOCK = RLock()
_ACTIVE = {}
_INSTALLED = set()


@contextmanager
def task_overview(span, task, result):
    from observability import set_span_input, set_span_output
    with capture_task(span) as captured:
        try:
            yield
        except BaseException as error:
            result.setdefault("error", str(error))
            result.setdefault("scheduler_status", "INTERRUPTED")
            raise
        finally:
            try:
                summary, inputs = build_overview(task, result, captured["rows"],
                                                 capture_limited=captured["limited"])
                set_span_input(span, inputs)
                set_span_output(span, summary)
            except Exception:
                # Presentation must not change execution, grading, or exception semantics.
                import logging
                logging.getLogger("agent").exception("Trace overview unavailable; original child spans remain")


@contextmanager
def capture_task(span, *, compact=False):
    """Capture only a live task's child spans; no background unbounded history."""
    rows = []
    capture = {"rows": rows, "limited": False, "compact": compact}
    if span is None:
        yield capture
        return
    from opentelemetry import trace
    from opentelemetry.sdk.trace import SpanProcessor
    provider = trace.get_tracer_provider()
    class CaptureProcessor(SpanProcessor):
        def on_end(self, finished):
            with _LOCK:
                sink = _ACTIVE.get(finished.context.trace_id)
                if sink is None:
                    return
                if len(sink["rows"]) >= 1000:
                    sink["limited"] = True
                    return
                attributes = dict(finished.attributes or {})
                if sink["compact"]:
                    # Normal conversations may run for hours. The overview
                    # needs a small index, not another in-memory copy of every
                    # prompt, tool result, and prior conversation turn.
                    compacted = {key: attributes[key] for key in (
                        "openinference.span.kind", "runtime.model_role",
                        "timing.duration_ms") if key in attributes}
                    if compacted.get("openinference.span.kind") == "LLM":
                        output = decoded(attributes.get("output.value"))
                        usage = output.get("usage") if isinstance(output, dict) else None
                        if isinstance(usage, dict):
                            cost = usage.get("cost")
                            compacted["trace_overview.usage"] = {
                                key: usage.get(key) for key in (
                                    "input_tokens", "output_tokens", "cache_read_tokens")
                            }
                            if isinstance(cost, dict):
                                compacted["trace_overview.usage"]["cost_cny"] = cost.get("total_cny")
                    if finished.name == "Scheduler / Dispatch Step":
                        output = decoded(attributes.get("output.value"))
                        if isinstance(output, dict):
                            step = _plain(output.get("current_step"))
                            compacted["trace_overview.step"] = {
                                "current_step": {"step_id": step.get("step_id"),
                                                 "worker_kind": step.get("worker_kind")},
                                "current_step_attempt": output.get("current_step_attempt"),
                            }
                    attributes = compacted
                sink["rows"].append({"name": finished.name,
                    "span_id": format(finished.context.span_id, "016x"),
                    "parent_id": format(finished.parent.span_id, "016x") if finished.parent else None,
                    "start_time": finished.start_time,
                    "attributes": attributes})
    with _LOCK:
        if id(provider) not in _INSTALLED and hasattr(provider, "add_span_processor"):
            provider.add_span_processor(CaptureProcessor())
            _INSTALLED.add(id(provider))
        trace_id = span.get_span_context().trace_id
        _ACTIVE[trace_id] = capture
    try:
        yield capture
    finally:
        with _LOCK:
            _ACTIVE.pop(trace_id, None)
