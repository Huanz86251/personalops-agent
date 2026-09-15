"""Read-only trace presentation. Never feeds summaries back into an agent."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from threading import RLock


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
def capture_task(span):
    """Capture only a live task's child spans; no background unbounded history."""
    rows = []
    capture = {"rows": rows, "limited": False}
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
                sink["rows"].append({"name": finished.name,
                    "span_id": format(finished.context.span_id, "016x"),
                    "parent_id": format(finished.parent.span_id, "016x") if finished.parent else None,
                    "attributes": dict(finished.attributes or {})})
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
