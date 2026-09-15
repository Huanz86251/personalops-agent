import json
import unittest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from config import PlanningSettings
from eventing import AsyncEventStore
from planning_graph import build_planning_graph
from planning_models import PlanningContextPack
from workers import WorkerAgentRegistry, WorkerGroupCoordinator
from workers.code_state import CodeRuntimeContextMiddleware


class RepairAwareWorker:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, input_state, config=None):
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        self.calls.append({
            "thread_id": thread_id,
            "repair": input_state.get("final_worker_repair_request"),
            "model_limit": input_state.get("executor_model_run_limit"),
            "tool_limit": input_state.get("executor_tool_run_limit"),
        })
        repaired = bool(input_state.get("final_worker_repair_request"))
        return {
            "messages": [
                *input_state.get("messages", []),
                AIMessage(content="已补齐。" if repaired else "首次提交。"),
            ],
            "executor_model_calls_used": 1,
            "executor_tool_calls_used": 1,
            "show_all_toolsets_calls_used": 0,
            "general_result": {
                "status": "COMPLETED",
                "summary": "已补齐最终缺口。" if repaired else "首次执行完成。",
                "criterion_claims": [{
                    "criterion_id": "C1",
                    "criterion": "完成并核验",
                    "evidence_tool_call_ids": [],
                    "conclusion": "已完成。",
                }],
                "unresolved_items": [],
                "evidence_tool_call_ids": [],
            },
        }


class FinalLoopModel:
    def __init__(self):
        self.final_calls = 0

    def with_structured_output(self, schema, **kwargs):
        async def respond(messages):
            if schema.__name__ == "SupervisorDecision":
                value = {
                    "action": "PLAN",
                    "plan_objective": "完成任务",
                    "plan_success_criteria": ["完成并核验"],
                    "steps": [{
                        "step_id": 1,
                        "worker_kind": "GENERAL",
                        "objective": "完成任务",
                        "success_criteria": ["完成并核验"],
                    }],
                }
            elif schema.__name__ == "FinalReviewDecision":
                self.final_calls += 1
                if self.final_calls == 1:
                    value = {
                        "review_reason": "缺少最终收尾证据。",
                        "criterion_reviews": [{
                            "criterion_id": "C1",
                            "evidence_refs": [],
                            "observed_result": "尚未观察到最终收尾证据。",
                            "missing_requirement": "缺少最终收尾证据。",
                            "status": "NOT_MET",
                        }],
                        "repair_request": {
                            "step_id": 1,
                            "worker_kind": "GENERAL",
                            "failed_criterion_ids": ["C1"],
                            "evidence_refs": [],
                            "observed_problem": "尚未观察到最终收尾证据。",
                            "missing_requirement": "缺少最终收尾证据。",
                        },
                        "replan_reason": None,
                        "action": "RETURN_TO_WORKER",
                        "status": None,
                        "final_answer": None,
                        "unmet_success_criteria": [],
                    }
                else:
                    value = {
                        "review_reason": "返修后证据完整。",
                        "criterion_reviews": [{
                            "criterion_id": "C1",
                            "evidence_refs": [],
                            "observed_result": "最后Worker已重新提交完整结果。",
                            "missing_requirement": None,
                            "status": "MET",
                        }],
                        "repair_request": None,
                        "replan_reason": None,
                        "action": "FINAL",
                        "status": "COMPLETED",
                        "final_answer": "已完成。",
                        "unmet_success_criteria": [],
                    }
            else:
                raise AssertionError(schema.__name__)
            return {
                "parsed": schema.model_validate(value),
                "raw": AIMessage(content=json.dumps(value, ensure_ascii=False)),
                "parsing_error": None,
            }
        return RunnableLambda(respond)


def settings():
    return PlanningSettings(
        max_steps_per_plan=2,
        max_total_steps=3,
        max_replans=1,
        max_step_model_rounds=8,
        max_step_executor_rounds=5,
        max_step_report_rounds=2,
        max_step_tool_calls=5,
        max_step_attempts=2,
        max_plan_model_rounds=14,
        max_plan_tool_calls=8,
        hard_recent_dialogue_turns=4,
        hard_recent_dialogue_max_chars=8000,
        conversation_summary_trigger_turns=8,
        conversation_summary_max_chars=4000,
        executor_summary_trigger_tokens=5000,
        executor_summary_trigger_messages=18,
        executor_summary_keep_messages=10,
        max_final_worker_repair_rounds=3,
    )


class FinalWorkerRepairLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_code_worker_and_reviewer_both_receive_final_repair_request(self):
        request = {
            "step_id": 2,
            "worker_kind": "CODE",
            "failed_criterion_ids": ["C2"],
            "evidence_refs": ["T7"],
            "observed_problem": "发布回执缺失。",
            "missing_requirement": "需要可核验的发布回执。",
        }
        state = {
            "messages": [],
            "code_task": {"task_id": "task"},
            "final_worker_repair_request": request,
        }
        for role in ("WORKER", "REVIEWER"):
            update = CodeRuntimeContextMiddleware(role).before_model(
                state,
                runtime=None,
            )
            self.assertIsNotNone(update)
            rendered = update["messages"][0].content
            self.assertIn("Final Reviewer事实型返修单", rendered)
            self.assertIn("发布回执缺失", rendered)

    async def test_returns_to_same_checkpoint_with_fresh_normal_budget(self):
        async with AsyncEventStore() as store:
            worker = RepairAwareWorker()
            model = FinalLoopModel()
            graph = build_planning_graph(
                simple_model=model,
                hard_model=model,
                worker_registry=WorkerAgentRegistry({"GENERAL": worker}),
                worker_group_coordinator=WorkerGroupCoordinator(store),
                planning=settings(),
                model_output_max_tokens=2048,
            )
            async def progress(_event):
                return None
            result = await graph.ainvoke({
                "context": PlanningContextPack(
                    current_time="now",
                    user_request="完成任务",
                    skill_mode="off",
                ),
                "event_id": "evt",
                "planning_run_id": "run",
                "conversation_thread_id": "conversation",
            }, config={"configurable": {"progress_callback": progress}})

        self.assertEqual(result["final_status"], "COMPLETED")
        self.assertEqual(model.final_calls, 2)
        self.assertEqual(len(worker.calls), 2)
        self.assertEqual(worker.calls[0]["thread_id"], worker.calls[1]["thread_id"])
        self.assertIsNone(worker.calls[0]["repair"])
        self.assertEqual(worker.calls[1]["repair"]["failed_criterion_ids"], ["C1"])
        self.assertEqual(worker.calls[1]["model_limit"], settings().max_step_executor_rounds)
        self.assertEqual(worker.calls[1]["tool_limit"], settings().max_step_tool_calls)
        self.assertEqual(result["final_worker_repair_round"], 1)
        self.assertEqual(
            result["current_step_trace"]["final_reviewer_request"]["observed_problem"],
            "尚未观察到最终收尾证据。",
        )


if __name__ == "__main__":
    unittest.main()
