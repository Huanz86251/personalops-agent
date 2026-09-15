"""Provider-free checks of conversation continuity and exact protected payloads."""
import copy
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from hard_planning import run_hard_supervisor, run_hard_replanner, run_hard_final_reviewer, run_hard_worker_leader
from planning_models import PlanningContextPack, PlanStep, StepReport
from scheduler_runtime import (
    SchedulerConversation,
    CURRENT_SCHEDULER,
    compact_schema,
    record_graph_facts,
    update_active_plan,
)
from workers.leadership_models import LeadershipWakeRequest, LeadershipWorkerView
from workers.compaction import WorkerCompactionMiddleware
from workers.submission import resolve_tool_evidence


class ScriptedModel:
    def __init__(self):
        self.requests = []

    def with_structured_output(self, schema, **kwargs):
        async def run(messages):
            self.requests.append(copy.deepcopy(messages))
            outputs = {
                "SupervisorDecision": {"action": "PLAN", "plan_objective": "海报", "plan_success_criteria": ["海报包含两家影院"],
                                       "steps": [{"step_id": 1, "objective": "查影院", "success_criteria": ["确认地址"], "worker_kind": "WEB"}]},
                "ReplanDecision": {"action": "FINISH", "reason": "已有结果", "remaining_steps": []},
                "FinalReviewDecision": {"action": "FINAL", "status": "PARTIAL", "final_answer": "保留未确认项"},
                "LeadershipDecision": {"action": "GUIDE", "reason": "补来源", "target_worker_ids": ["web1"], "guidance": "核对地址"},
            }
            payload = outputs[schema.__name__]
            return {"parsed": schema.model_validate(payload), "raw": AIMessage(content=json.dumps(payload)), "parsing_error": None}
        return RunnableLambda(run)


class ProgressiveSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_once_and_every_request_extends_previous_history(self):
        model = ScriptedModel()
        context = PlanningContextPack(current_time="now", user_request="查影院并做海报")
        await run_hard_supervisor(model, context=context, max_steps_per_plan=2)
        report = StepReport(step_id=1, status="PARTIAL", summary="影院甲地址待核实", stop_reason="来源不足")
        for reason in ("先停止", "用户要求保留文件名 poster.png"):
            await run_hard_replanner(model, context=context, plan_objective="海报", plan_success_criteria=["海报包含两家影院"],
                                    completed_step_reports=[report], replan_context=reason, remaining_steps=[],
                                    remaining_budget={"model": 3}, next_step_id=2, max_remaining_steps=1)
        await run_hard_final_reviewer(model, context=context, plan_objective="海报", plan_success_criteria=["海报包含两家影院"],
                                     step_reports=[report], replan_history=[], overall_stop_reason="结束", replan_available=False)
        # Supervisor and Replanner share the Scheduler identity prefix.
        # Final Reviewer is deliberately a separate minimal evidence call.
        for request in model.requests[:3]:
            self.assertEqual(request[0], model.requests[0][0])
        self.assertEqual(len({request[0]["content"] for request in model.requests[:3]}), 1)
        self.assertNotEqual(model.requests[-1][0], model.requests[0][0])
        protocols = [record for record in context.scheduler_session["records"] if record["kind"] == "protocol"]
        self.assertEqual(len(protocols), 1)
        self.assertIn("replanner", protocols[0]["key"])
        self.assertEqual(sum(record["kind"] == "StepReport" for record in context.scheduler_session["records"]), 1)
        self.assertFalse(any(
            str(record.get("key", "")).startswith("transient:")
            for record in context.scheduler_session["records"]
        ))
        self.assertEqual(
            context.scheduler_session["active_records"]["活动计划"]["kind"],
            "活动计划",
        )
        self.assertNotIn("FinalReviewDecision", json.dumps(model.requests[0]))
        restored = PlanningContextPack.model_validate_json(context.model_dump_json())
        self.assertEqual(restored.scheduler_session, context.scheduler_session)

    async def test_progress_uses_same_scheduler_and_keeps_guidance(self):
        model = ScriptedModel()
        context = PlanningContextPack(current_time="now", user_request="查影院")
        await run_hard_supervisor(model, context=context, max_steps_per_plan=1)
        session = SchedulerConversation(context.scheduler_session, context)
        token = CURRENT_SCHEDULER.set(session)
        try:
            request = LeadershipWakeRequest(wake_id="wake1", event_id="event1", reason="WORKER_BLOCKED", created_at=datetime.now(timezone.utc),
                workers=[LeadershipWorkerView(worker_id="web1", assignment="查地址", cursor_before=0, reports=[{"summary": "缺来源"}])])
            await run_hard_worker_leader(model, request=request)
        finally:
            CURRENT_SCHEDULER.reset(token)
        self.assertEqual(model.requests[0][0], model.requests[1][0])
        self.assertIn(
            "核对地址",
            context.scheduler_session["active_records"]["当前Worker控制"]["content"],
        )
        self.assertFalse(any(
            str(record.get("key", "")).startswith("transient:")
            for record in context.scheduler_session["records"]
        ))

    def test_active_plan_is_replaced_in_place_and_is_prompt_tail(self):
        context = PlanningContextPack(current_time="now", user_request="查影院")
        session = SchedulerConversation(context.scheduler_session, context)
        session.initialize()
        session.set_active("活动计划", {"version": 1, "steps": ["旧步骤"]})
        session.set_active("当前Worker控制", {"action": "GUIDE"}, role="assistant")
        session.set_active("活动计划", {"version": 2, "steps": ["新步骤"]})

        wire = session.wire()
        self.assertIn('"version":2', wire[-1]["content"])
        self.assertNotIn("旧步骤", json.dumps(wire, ensure_ascii=False))
        self.assertEqual(
            list(context.scheduler_session["active_records"]),
            ["活动计划", "当前Worker控制"],
        )

    def test_reviewed_step_contract_and_delivery_interfaces_survive_compaction(self):
        context = PlanningContextPack(current_time="now", user_request="交付文件")
        session = SchedulerConversation(context.scheduler_session, context, threshold=1)
        session.initialize()
        step = PlanStep(
            step_id=1,
            objective="生成并交付报告",
            success_criteria=["报告发布到约定路径"],
            worker_kind="WEB",
            artifact_outputs=[{
                "output_id": "report",
                "description": "用户报告",
                "disposition": "USER_DELIVERABLE",
                "target_path": "reports/final.md",
            }],
        )
        report = StepReport(
            step_id=1,
            status="COMPLETED",
            summary="报告已验收",
            stop_reason="独立审核完成",
            artifacts=[{"path": "/handoff/reports/final.md", "description": "已发布"}],
        )
        state = {
            "plan_objective": "交付报告",
            "plan_success_criteria": ["报告发布到约定路径"],
            "current_step": step,
            "remaining_steps": [],
            "completed_step_reports": [report],
            "handoff_publication_receipts": [{
                "output_id": "report",
                "target_path": "reports/final.md",
                "handoff_path": "/handoff/reports/final.md",
                "storage_path": "D:/private/run/reports/final.md",
            }],
        }
        record_graph_facts(session, state)
        update_active_plan(session, state)
        session.add("protocol", "可重建协议" * 100, protected=False, key="old-protocol")
        session.compact()

        durable = json.dumps(session.records, ensure_ascii=False)
        self.assertIn("已审核步骤契约", durable)
        self.assertIn("reports/final.md", durable)
        self.assertIn("/handoff/reports/final.md", durable)
        self.assertIn("StepReport", durable)
        self.assertFalse(any(record.get("key") == "old-protocol" for record in session.records))
        active_plan = session.active_records["活动计划"]["content"]
        self.assertIn('"step_id":1,"status":"COMPLETED"', active_plan)
        self.assertIn('"remaining_steps":[]', active_plan)

    def test_compaction_never_changes_plan_report_or_instruction(self):
        context = PlanningContextPack(current_time="now", user_request="原文不改")
        session = SchedulerConversation(context.scheduler_session, context, threshold=1)
        session.initialize()
        session.add("protocol", "旧协议", protected=False, key="old")
        session.fact("计划", {"path": "海报/最终.png", "steps": ["搜索", "制作"]})
        session.fact("StepReport", {"files": [{"path": "data.csv", "sha256": "abc"}], "unresolved": ["未核实"]})
        session.add("decision", '{"guidance":"不要使用未核实地址"}', role="assistant")
        exact = [copy.deepcopy(record) for record in session.records if record["protected"]]
        for index in range(20):
            session.add("progress", "重复过程" * 100, protected=False)
        session.compact()
        self.assertEqual(exact, [record for record in session.records if record["protected"]])
        self.assertFalse(any(record.get("key") == "old" for record in session.records))
        key = session.disclose("replanner", {"type": "object"})
        session.disclose("replanner", {"type": "object"})
        self.assertEqual(sum(record.get("key") == key for record in session.records), 1)

    def test_schema_compaction_keeps_property_names_and_constraints(self):
        schema = {"title": "Large display name", "type": "object", "properties": {"description": {"type": "string", "minLength": 1}},
                  "required": ["description"], "additionalProperties": False}
        compact = compact_schema(schema)
        self.assertIn("description", compact["properties"])
        self.assertEqual(compact["required"], ["description"])
        self.assertFalse(compact["additionalProperties"])

    def test_worker_compaction_defaults_keep_two_messages_and_four_thousand_tokens(self):
        with patch.dict(
            "os.environ",
            {
                "WORKER_SUMMARY_KEEP_MESSAGES": "2",
                "WORKER_SUMMARY_KEEP_TOKENS": "4000",
                "WORKER_COMPACTION_ENABLED": "false",
            },
            clear=False,
        ):
            middleware = WorkerCompactionMiddleware(None)
        self.assertEqual(middleware.keep_messages, 2)
        self.assertEqual(middleware.keep_tokens, 4000)
        self.assertFalse(middleware.enabled)

    async def test_worker_compaction_is_disabled_by_default(self):
        model = SimpleNamespace(ainvoke=AsyncMock())
        middleware = WorkerCompactionMiddleware(model, threshold=1)
        state = {
            "messages": [HumanMessage(content="keep this message")],
            "executor_model_run_limit": 6,
        }
        self.assertIsNone(await middleware.abefore_model(state, None))
        model.ainvoke.assert_not_awaited()

    async def test_worker_compression_keeps_tool_evidence_and_task(self):
        original = HumanMessage(content="原始任务原文")
        call = AIMessage(content="", tool_calls=[{"id": "search1", "name": "web_search", "args": {}}])
        result = ToolMessage(content="真实结果" * 1000, tool_call_id="search1", name="web_search")
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="已经搜索，需核对地址")))
        middleware = WorkerCompactionMiddleware(
            model, threshold=1, keep_messages=2, keep_tokens=10, enabled=True
        )
        state = {"messages": [original, call, result, *[HumanMessage(content=f"任务补充 {i}") for i in range(12)]],
                 "executor_model_run_limit": 6}
        update = await middleware.abefore_model(state, None)
        kept = update["messages"][1:]
        self.assertIn(original, kept)
        evidence = resolve_tool_evidence(update["worker_archived_messages"] + kept, ["search1"])
        self.assertEqual(evidence[0].tool_call_id, "search1")
        self.assertEqual(update["worker_compaction_calls_used"], 1)


if __name__ == "__main__":
    unittest.main()
