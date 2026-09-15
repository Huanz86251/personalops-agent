"""No provider calls: Code Worker budget -> real structured handoff."""
import json
import unittest
from types import SimpleNamespace
from langchain_core.messages import AIMessage, ToolMessage
from middlewares import DynamicExecutionBudgetMiddleware
from test_code_agents import ToolRecordingModel, candidate, contract, code_probe
from workers.code_worker import create_code_worker
from workers.code_review_models import create_code_review_loop
from workers.code_finalization import CodeWorkerBudgetMiddleware, terminal_tool, missing_handoff_message


def submission():
    return AIMessage(content="", tool_calls=[{"name": "submit_code_for_review", "id": "report",
        "args": {"submission": {"candidate": candidate().model_dump(mode="json"),
            "summary": "Stopped at budget; implementation unfinished.",
            "requirement_status": {"feature_works": "NOT_MET"},
            "limitations": ["Need Scheduler guidance; no passing test evidence."]}}}])


class CodeFinalizationTests(unittest.TestCase):
    def run_worker(self, responses, model_limit, tool_limit):
        model = ToolRecordingModel(responses=responses)
        graph = create_code_worker(model, tools=[code_probe], middleware=[DynamicExecutionBudgetMiddleware()])
        result = graph.invoke({"messages": [{"role": "user", "content": "Implement feature"}],
            "code_task": contract().model_dump(mode="json"), "code_candidate": candidate().model_dump(mode="json"),
            "code_review_loop": create_code_review_loop(candidate=candidate(), worker_checkpoint_id="w", reviewer_checkpoint_id="r").model_dump(mode="json"),
            "executor_model_run_limit": model_limit, "executor_tool_run_limit": tool_limit,
            "show_all_toolsets_run_limit": 0})
        return model, result

    def test_last_round_submits_not_met_without_spending_review_reserve(self):
        model, result = self.run_worker([submission()], 1, 0)
        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(model.bound_tool_names, ["submit_code_for_review"])
        self.assertEqual(result["executor_model_calls_used"], 1)
        self.assertEqual(result["code_worker_submission"]["submission"]["requirement_status"], {"feature_works": "NOT_MET"})
        self.assertNotIn("code_review_report", result)

    def test_business_tool_limit_triggers_submission(self):
        probe = AIMessage(content="", tool_calls=[{"name": "code_probe", "id": "p", "args": {"value": "probe"}}])
        model, result = self.run_worker([probe, submission()], 5, 1)
        self.assertEqual(model.invocation_count, 2)
        self.assertEqual(result["executor_model_calls_used"], 2)
        self.assertEqual(result["executor_tool_calls_used"], 1)
        self.assertEqual(result["worker_finalization_model_calls_used"], 1)

    def test_zero_budget_does_not_call_model(self):
        model, result = self.run_worker([], 0, 0)
        self.assertEqual(model.invocation_count, 0)
        self.assertFalse(result.get("code_worker_submission"))

    def test_natural_exit_still_gets_one_submission_round(self):
        model, result = self.run_worker([AIMessage(content="Unable to finish implementation."), submission()], 3, 3)
        self.assertEqual(model.invocation_count, 2)
        self.assertTrue(result.get("code_worker_submission"))
        self.assertEqual(result["executor_model_calls_used"], 2)

    def test_control_selection_follows_current_instruction(self):
        self.assertEqual(terminal_tool({"code_review_loop": {"scheduler_epoch": 1}}), "submit_code_for_review")
        self.assertEqual(terminal_tool({"code_review_loop": {"pending_instruction": {"round_no": 1}}}), "respond_to_code_review")
        self.assertEqual(terminal_tool({"code_review_loop": {"pending_scheduler_directive": {"scheduler_epoch": 2}}}), "submit_continued_code_for_review")

    def test_no_repeated_summary_and_no_business_calls_in_summary(self):
        gate = CodeWorkerBudgetMiddleware()
        state = {"worker_finalize_requested": True, "executor_model_run_limit": 5,
                 "worker_finalization_model_calls_used": 1}
        self.assertEqual(gate.before_model(state, None), {"jump_to": "end"})
        state["messages"] = [AIMessage(content="", tool_calls=[{"name":"code_probe","id":"bad","args":{"value":"bad"}}])]
        update = gate.after_model(state, None)
        self.assertEqual(update["messages"][-1].tool_calls, [])

    def test_diagnostic_includes_actual_error_and_usage(self):
        data = json.loads(missing_handoff_message({"execution_summary": {"model_call_count": 2},
            "current_turn_messages": [ToolMessage(content="NameError: missing input", status="error", tool_call_id="x")]}, "INITIAL_SUBMISSION"))
        self.assertIn("NameError", data["tool_errors"][0])
        self.assertEqual(data["usage"]["model_call_count"], 2)
        self.assertIn("not completed", data["reason"])

class CodeReviewerFinalizationTests(unittest.TestCase):
    def test_reviewer_last_round_only_submits_review(self):
        from workers.code_finalization import CodeReviewerBudgetMiddleware, CodeReviewerProgressMiddleware
        gate = CodeReviewerBudgetMiddleware()
        state = {'executor_model_run_limit':5, 'executor_model_calls_used':4}
        self.assertTrue(gate.before_model(state,None)['worker_finalize_requested'])
        state.update(worker_finalize_requested=True,messages=[AIMessage(content='',tool_calls=[
            {'name':'submit_code_review','id':'r','args':{}},
            {'name':'execute','id':'x','args':{}}])])
        update=gate.after_model(state,None)
        self.assertEqual([c['name'] for c in update['messages'][-1].tool_calls],['submit_code_review'])
        self.assertEqual(CodeReviewerProgressMiddleware.terminal_name({}), 'submit_code_review')
