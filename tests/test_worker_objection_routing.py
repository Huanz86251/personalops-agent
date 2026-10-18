"""Exercise Scheduler adjudication -> existing global replan, without providers."""
from types import SimpleNamespace

import pytest
import json
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from tests import test_code_global_replan as routing_fixture
from tests.test_planning_code_execution import AppliedCodeRuntime
from tests.test_worker_plan_objection import offline_async
from schemas.worker_control import WorkerPlanObjection


class ObjectionModel(routing_fixture.ReplanModel):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.objection_inputs = []
        self.final_control_packets = []

    def with_structured_output(self, schema, **kwargs):
        if schema.__name__ == "FinalReviewDecision":
            delegate = super().with_structured_output(schema, **kwargs)
            async def final_reply(messages):
                self.final_control_packets.append(json.loads(messages[1]["content"])["独立审核材料"])
                return await delegate.ainvoke(messages)
            return RunnableLambda(final_reply)
        if schema.__name__ == "WorkerObjectionDecision":
            def reply(messages):
                self.objection_inputs.append(messages)
                payload = {"action": self.action, "reason": "Plan uses the wrong input source.",
                           "guidance": "Keep the assigned contract and use the CSV." if self.action == "GUIDE" else None}
                return {"parsed": schema.model_validate(payload), "raw": AIMessage(content="control"), "parsing_error": None}
            return RunnableLambda(reply)
        return super().with_structured_output(schema, **kwargs)


class ObjectionRuntime(AppliedCodeRuntime):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.decisions = []

    async def ainvoke(self, state, config=None):
        self.calls += 1
        if self.calls == 1:
            handler = config["configurable"]["worker_objection_handler"]
            objection = WorkerPlanObjection(problem="Wrong input source.", reason="User supplied CSV.",
                                           suggested_direction="Read CSV instead of database.")
            result = await handler(objection, state)
            self.decisions.append(result)
            if result["decision"]["action"] in {"REPLAN", "FINISH"}:
                result = {"objection": objection.model_dump(mode="json"), **result}
                return {"messages": [AIMessage(content="Paused for Scheduler.")],
                        "executor_model_calls_used": 1, "executor_tool_calls_used": 0,
                        "worker_leadership_model_rounds_used": result["model_rounds_used"],
                        "worker_objection_outcome": result, "worker_objection_records": [result]}
        return await super().ainvoke(state, config=config)


@pytest.mark.parametrize("action", ["CONTINUE", "GUIDE", "REPLAN", "FINISH"])
@offline_async
async def test_scheduler_decisions_use_existing_graph_routes(action):
    model, runtime = ObjectionModel(action), ObjectionRuntime()
    result = await routing_fixture.CodeGlobalReplanTests().run_graph(model, runtime)
    assert len(model.objection_inputs) == 1
    assert model.step_report_calls == 0
    assert len(model.replan_inputs) == (1 if action == "REPLAN" else 0)
    assert result["replans_used"] == (1 if action == "REPLAN" else 0)
    assert runtime.calls == (2 if action == "REPLAN" else 1)
    if action in {"REPLAN", "FINISH"}:
        report = result["completed_step_reports"][0]
        assert report.assessment_source == "SCHEDULER_CONTROL"
        assert report.status == "BLOCKED"
        assert report.next_action == "Read CSV instead of database."
    if action == "FINISH":
        assert model.final_control_packets[-1]["return_to_worker_available"] is False
    assert "final_answer" in result


@offline_async
async def test_no_replan_quota_guides_current_worker_instead():
    model, runtime = ObjectionModel("REPLAN"), ObjectionRuntime()
    result = await routing_fixture.CodeGlobalReplanTests().run_graph(model, runtime, max_replans=0)
    assert runtime.decisions[0]["decision"]["action"] == "GUIDE"
    assert len(model.replan_inputs) == 0
    assert runtime.calls == 1
    assert result["replans_used"] == 0
