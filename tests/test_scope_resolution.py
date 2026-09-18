import unittest
import os
from unittest.mock import AsyncMock, patch

from planning_models import PlanningContextPack, ScopeContract, SupervisorDecision
from hard_planning import _validate_supervisor_scope_contract
from scheduler_runtime import SchedulerConversation
from scope_resolution import prepare_scope_context


CONTRACT = ScopeContract.model_validate({
    "target_entity": "bookmark",
    "effect_mode": "MUTATION",
    "constraints": [
        {"source_text": "工作标签", "applies_to": "bookmark", "meaning": "书签带有工作标签"},
        {"source_text": "尚未归档", "applies_to": "bookmark", "meaning": "书签不在归档区"},
    ],
    "sets": [
        {"set_id": "A", "definition": "工作标签书签", "result_entity": "bookmark"},
        {"set_id": "B", "definition": "尚未归档书签", "result_entity": "bookmark"},
    ],
    "operation": "INTERSECTION",
    "operands": ["A", "B"],
    "join_key": "bookmark_id",
    "ambiguity": False,
    "alternatives": [],
    "required_context": [],
})


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.options = kwargs
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages = messages
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class SequencedStructuredModel(FakeStructuredModel):
    def __init__(self, responses):
        super().__init__(None)
        self.responses = list(responses)
        self.message_history = []

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages = messages
        self.message_history.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class LengthFinishReasonError(ValueError):
    pass


def context(text="把工作标签且尚未归档的书签移到归档区"):
    return PlanningContextPack(current_time="now", user_request=text)


class ScopeResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_request_stays_local(self):
        model = FakeStructuredModel(AssertionError("cloud model must not run"))
        route = {
            "label": "DIRECT_RESPONSE", "label_id": 0,
            "probability_requires_scope": 0.01, "threshold": 0.1,
            "router_status": "ok",
        }
        with patch("scope_resolution.classify_scope_request", AsyncMock(return_value=route)):
            updated, calls = await prepare_scope_context(context("你好"), model)
        self.assertEqual(calls, 0)
        self.assertIsNone(updated.scope_contract)
        self.assertEqual(updated.scope_router["label"], "DIRECT_RESPONSE")
        self.assertEqual(model.calls, 0)

    async def test_positive_route_generates_and_freezes_contract(self):
        model = FakeStructuredModel({"parsed": CONTRACT, "parsing_error": None})
        route = {
            "label": "REQUIRES_SCOPE_CONTRACT", "label_id": 1,
            "probability_requires_scope": 0.99, "threshold": 0.1,
            "router_status": "ok",
        }
        with patch("scope_resolution.classify_scope_request", AsyncMock(return_value=route)):
            updated, calls = await prepare_scope_context(context(), model)
            resumed, resumed_calls = await prepare_scope_context(updated, model)
        self.assertEqual(calls, 1)
        self.assertEqual(resumed_calls, 0)
        self.assertEqual(model.calls, 1)
        self.assertEqual(resumed.scope_contract.operation, "INTERSECTION")
        self.assertEqual(model.options["method"], "json_mode")
        self.assertIn('"target_entity"', model.messages[0]["content"])
        self.assertIn("bookmark_id", model.messages[0]["content"])
        self.assertIn("真实世界当前时间", model.messages[1]["content"])
        self.assertIn("now", model.messages[1]["content"])

    def test_contract_preserves_cross_entity_required_context_and_read_only_effect(self):
        contract = ScopeContract.model_validate({
            "target_entity": "movie_title",
            "effect_mode": "READ_ONLY",
            "constraints": [{
                "source_text": "from my Simple Note account",
                "applies_to": "movie_title",
                "meaning": "电影候选来自用户的Simple Note",
            }],
            "sets": [{
                "set_id": "A",
                "definition": "Simple Note中的候选电影标题",
                "result_entity": "movie_title",
            }],
            "operation": "DIRECT",
            "operands": ["A"],
            "join_key": None,
            "ambiguity": False,
            "alternatives": [],
            "required_context": [{
                "read": "Laura发来的电影推荐短信",
                "source_system": "messaging",
                "relationship": "null",
                "resolution_status": "EXPLICIT",
                "resolution_note": None,
                "because": "短信中可能包含电影筛选要求",
                "used_for": "从集合A中筛出最终回复内容",
            }],
        })
        self.assertEqual(contract.effect_mode, "READ_ONLY")
        self.assertEqual(contract.required_context[0].read, "Laura发来的电影推荐短信")
        self.assertEqual(contract.required_context[0].source_system, "messaging")
        self.assertIsNone(contract.required_context[0].relationship)
        schema = ScopeContract.model_json_schema()["properties"]
        self.assertLess(list(schema).index("effect_mode"), list(schema).index("ambiguity"))
        self.assertIn("required_context", schema)

    def test_personal_relationship_context_binds_authoritative_source(self):
        contract = ScopeContract.model_validate({
            "target_entity": "venmo_payment_request",
            "effect_mode": "MUTATION",
            "constraints": [{
                "source_text": "from my friends and roommates",
                "applies_to": "request sender",
                "meaning": "发送者是Phone联系人中的friend或roommate",
            }],
            "sets": [{
                "set_id": "A",
                "definition": "来自Phone联系人friend或roommate的待处理Venmo请求",
                "result_entity": "venmo_payment_request",
            }],
            "operation": "DIRECT",
            "operands": ["A"],
            "required_context": [{
                "read": "Phone联系人中的朋友",
                "source_system": "phone contact book",
                "relationship": "friend",
                "resolution_status": "EXPLICIT",
                "resolution_note": None,
                "because": "需要确定哪些发送者属于用户的朋友",
                "used_for": "筛选待处理Venmo请求的发送者",
            }, {
                "read": "Phone联系人中的室友",
                "source_system": "phone contact book",
                "relationship": "roommate",
                "resolution_status": "EXPLICIT",
                "resolution_note": None,
                "because": "需要确定哪些发送者属于用户的室友",
                "used_for": "筛选待处理Venmo请求的发送者",
            }],
        })
        self.assertEqual(
            [item.relationship for item in contract.required_context],
            ["friend", "roommate"],
        )
        self.assertTrue(all(
            item.source_system == "phone contact book"
            for item in contract.required_context
        ))

    def test_relative_time_contract_uses_absolute_ranges_and_normalizes_nullish_values(self):
        contract = ScopeContract.model_validate({
            "target_entity": "file",
            "effect_mode": "MUTATION",
            "constraints": [{
                "source_text": "今年3月",
                "applies_to": "file",
                "applies_to_sets": ["A"],
                "meaning": "创建日期位于2023-03-01至2023-03-31",
            }],
            "sets": [{
                "set_id": "A",
                "definition": "2023-03-01至2023-03-31创建的文件",
                "result_entity": "file",
            }],
            "operation": "DIRECT",
            "operands": ["A"],
            "join_key": "NULL",
            "ambiguity": False,
            "alternatives": False,
            "required_context": "false",
            "resolved_time_ranges": [{
                "source_text": "今年3月",
                "start_at": "2023-03-01",
                "end_at": "2023-03-31",
                "resolution_status": "DERIVED",
                "resolution_note": "由任务开始时间唯一换算",
                "used_for_sets": ["A"],
            }, {
                "source_text": "未定义财年",
                "start_at": "null",
                "end_at": False,
                "resolution_status": "UNKNOWN",
                "resolution_note": "缺少财年起始月",
            }],
        })
        self.assertIsNone(contract.join_key)
        self.assertEqual(contract.alternatives, [])
        self.assertEqual(contract.required_context, [])
        self.assertEqual(contract.resolved_time_ranges[0].start_at, "2023-03-01")
        self.assertEqual(contract.resolved_time_ranges[0].resolution_status, "DERIVED")
        self.assertIsNone(contract.resolved_time_ranges[1].start_at)
        self.assertIsNone(contract.resolved_time_ranges[1].end_at)
        self.assertEqual(contract.resolved_time_ranges[1].resolution_status, "UNKNOWN")
        schema = ScopeContract.model_json_schema()["properties"]
        self.assertIn("resolved_time_ranges", schema)

    def test_scope_relations_can_bind_known_facts_to_specific_sets(self):
        contract = ScopeContract.model_validate({
            "target_entity": "message",
            "effect_mode": "MUTATION",
            "constraints": [{
                "source_text": "未读",
                "applies_to": "message",
                "applies_to_sets": ["A", "B"],
                "meaning": "两个分支都必须是未读消息",
            }],
            "sets": [
                {"set_id": "A", "definition": "来自朋友的未读消息", "result_entity": "message"},
                {"set_id": "B", "definition": "来自室友的未读消息", "result_entity": "message"},
            ],
            "operation": "UNION",
            "operands": ["A", "B"],
            "join_key": "message_id",
            "required_context": [{
                "read": "朋友名单",
                "source_system": "phone contact book",
                "relationship": "friend",
                "resolution_status": "EXPLICIT",
                "used_for_sets": ["A"],
                "because": "需要确定朋友身份",
                "used_for": "筛选集合A",
            }],
        })
        self.assertEqual(contract.constraints[0].applies_to_sets, ["A", "B"])
        self.assertEqual(contract.required_context[0].used_for_sets, ["A"])
        invalid = contract.model_dump(mode="json")
        invalid["required_context"][0]["used_for_sets"] = ["MISSING"]
        with self.assertRaisesRegex(ValueError, "used_for_sets"):
            ScopeContract.model_validate(invalid)

    def test_resolution_status_rejects_unexplained_guess_and_unknown_time_values(self):
        base = {
            "target_entity": "message",
            "effect_mode": "READ_ONLY",
            "constraints": [],
            "sets": [{"set_id": "A", "definition": "目标消息", "result_entity": "message"}],
            "operation": "DIRECT",
            "operands": ["A"],
        }
        with self.assertRaisesRegex(ValueError, "resolution_note"):
            ScopeContract.model_validate({
                **base,
                "required_context": [{
                    "read": "朋友名单",
                    "source_system": "phone contact book",
                    "relationship": "friend",
                    "resolution_status": "INFERRED",
                    "because": "需要识别发送者",
                    "used_for": "筛选目标消息",
                }],
            })
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            ScopeContract.model_validate({
                **base,
                "resolved_time_ranges": [{
                    "source_text": "本财年",
                    "start_at": "2023-01-01",
                    "end_at": None,
                    "resolution_status": "UNKNOWN",
                    "resolution_note": "缺少财年起始月",
                }],
            })

    def test_scheduler_scope_tail_must_match_frozen_contract(self):
        target = {
            "target_entity": "bookmark",
            "effect_mode": "MUTATION",
            "constraints": [
                {"source_text": "工作标签", "applies_to": "bookmark",
                 "applies_to_sets": ["A"], "meaning": "书签带有工作标签"},
                {"source_text": "尚未归档", "applies_to": "bookmark",
                 "applies_to_sets": ["B"], "meaning": "书签不在归档区"},
            ],
            "sets": [
                {"set_id": "A", "definition": "工作标签书签",
                 "condition_owner": "bookmark", "result_entity": "bookmark"},
                {"set_id": "B", "definition": "尚未归档书签",
                 "condition_owner": "bookmark", "result_entity": "bookmark"},
            ],
            "operation": "INTERSECTION",
            "operands": ["A", "B"],
            "join_key": "bookmark_id",
            "required_context": [],
        }
        decision = SupervisorDecision.model_validate({
            "action": "PLAN",
            "plan_objective": "归档目标书签",
            "plan_success_criteria": ["目标书签已归档"],
            "steps": [{
                "step_id": 1,
                "objective": "处理目标书签",
                "success_criteria": ["目标书签已归档"],
                "target_selection": target,
                "worker_kind": "GENERAL",
            }],
        })
        self.assertIs(_validate_supervisor_scope_contract(decision, CONTRACT), decision)
        wrong = decision.model_copy(deep=True)
        wrong.steps[0].target_selection.effect_mode = "READ_ONLY"
        with self.assertRaisesRegex(ValueError, "effect_mode"):
            _validate_supervisor_scope_contract(wrong, CONTRACT)

    def test_scheduler_must_copy_resolved_time_ranges_exactly(self):
        contract = ScopeContract.model_validate({
            "target_entity": "file",
            "effect_mode": "MUTATION",
            "constraints": [],
            "sets": [{"set_id": "A", "definition": "今年3月的文件", "result_entity": "file"}],
            "operation": "DIRECT",
            "operands": ["A"],
            "join_key": None,
            "resolved_time_ranges": [{
                "source_text": "今年3月",
                "start_at": "2023-03-01",
                "end_at": "2023-03-31",
                "resolution_status": "DERIVED",
                "resolution_note": "由任务开始时间唯一换算",
            }],
        })
        target = {
            "target_entity": "file",
            "effect_mode": "MUTATION",
            "sets": [{
                "set_id": "A", "definition": "今年3月的文件",
                "condition_owner": "file", "result_entity": "file",
            }],
            "operation": "DIRECT",
            "operands": ["A"],
            "join_key": None,
            "resolved_time_ranges": [{
                "source_text": "今年3月",
                "start_at": "2023-03-01",
                "end_at": "2023-03-31",
                "resolution_status": "DERIVED",
                "resolution_note": "由任务开始时间唯一换算",
            }],
        }
        decision = SupervisorDecision.model_validate({
            "action": "PLAN",
            "plan_objective": "整理照片",
            "plan_success_criteria": ["照片按绝对日期归类"],
            "steps": [{
                "step_id": 1,
                "objective": "处理2023年3月照片",
                "success_criteria": ["仅处理2023-03-01至2023-03-31"],
                "target_selection": target,
                "worker_kind": "GENERAL",
            }],
        })
        self.assertIs(_validate_supervisor_scope_contract(decision, contract), decision)
        wrong = decision.model_copy(deep=True)
        wrong.steps[0].target_selection.resolved_time_ranges[0].start_at = "2022-03-01"
        with self.assertRaisesRegex(ValueError, "resolved_time_ranges"):
            _validate_supervisor_scope_contract(wrong, contract)

    def test_plan_step_normalizes_common_optional_sentinels(self):
        step = SupervisorDecision.model_validate({
            "action": "PLAN",
            "plan_objective": "直接处理",
            "plan_success_criteria": ["已处理"],
            "steps": [{
                "step_id": 1,
                "objective": "处理",
                "success_criteria": ["完成"],
                "rag_query": "NULL",
                "execution_guidance": False,
                "api_suggestion": "AULL",
                "target_selection": "false",
                "code_task": "none",
                "worker_kind": "GENERAL",
            }],
        }).steps[0]
        self.assertIsNone(step.rag_query)
        self.assertIsNone(step.execution_guidance)
        self.assertIsNone(step.api_suggestion)
        self.assertIsNone(step.target_selection)
        self.assertIsNone(step.code_task)

    async def test_resolver_failure_is_observable_without_hidden_retry(self):
        model = FakeStructuredModel(RuntimeError("provider failed"))
        route = {
            "label": "REQUIRES_SCOPE_CONTRACT", "label_id": 1,
            "probability_requires_scope": 0.9, "threshold": 0.1,
            "router_status": "ok",
        }
        with patch("scope_resolution.classify_scope_request", AsyncMock(return_value=route)):
            updated, calls = await prepare_scope_context(context(), model)
        self.assertEqual(calls, 1)
        self.assertEqual(model.calls, 1)
        self.assertIsNone(updated.scope_contract)
        self.assertEqual(updated.scope_router["resolver_status"], "failed")
        self.assertEqual(updated.scope_router["resolver_error_type"], "RuntimeError")

    async def test_schema_failure_is_repaired_by_same_resolver(self):
        invalid = {
            "raw": {"content": '{"target_entity":"bookmark"}'},
            "parsed": None,
            "parsing_error": ValueError("incomplete contract"),
        }
        model = SequencedStructuredModel([
            invalid,
            {"parsed": CONTRACT, "parsing_error": None},
        ])
        route = {
            "label": "REQUIRES_SCOPE_CONTRACT", "label_id": 1,
            "probability_requires_scope": 0.9, "threshold": 0.1,
            "router_status": "ok",
        }
        with patch.dict(os.environ, {"SCOPE_RESOLVER_MAX_ATTEMPTS": "3"}), patch(
            "scope_resolution.classify_scope_request", AsyncMock(return_value=route),
        ):
            updated, calls = await prepare_scope_context(context(), model)
        self.assertEqual(calls, 2)
        self.assertEqual(model.calls, 2)
        self.assertEqual(updated.scope_router["resolver_status"], "ok")
        repair = model.message_history[1][-1]["content"]
        self.assertIn("字段校验失败", repair)
        self.assertIn("sets: missing", repair)
        self.assertIn("operation: missing", repair)
        self.assertIn("只返修 ScopeContract", repair)

    async def test_truncation_requests_shorter_reasoning_and_complete_json(self):
        truncated = {
            "raw": {
                "content": "",
                "response_metadata": {"finish_reason": "length"},
            },
            "parsed": None,
            "parsing_error": ValueError("invalid json"),
        }
        model = SequencedStructuredModel([
            truncated,
            {"parsed": CONTRACT, "parsing_error": None},
        ])
        route = {
            "label": "REQUIRES_SCOPE_CONTRACT", "label_id": 1,
            "probability_requires_scope": 0.9, "threshold": 0.1,
            "router_status": "ok",
        }
        with patch.dict(os.environ, {"SCOPE_RESOLVER_MAX_ATTEMPTS": "3"}), patch(
            "scope_resolution.classify_scope_request", AsyncMock(return_value=route),
        ):
            updated, calls = await prepare_scope_context(context(), model)
        self.assertEqual(calls, 2)
        self.assertEqual(updated.scope_router["resolver_status"], "ok")
        repair = model.message_history[1][-1]["content"]
        self.assertIn("输出被截断", repair)
        self.assertIn("缩短思考过程", repair)
        self.assertIn("完整 JSON", repair)

    async def test_thrown_length_limit_error_uses_repair_round(self):
        model = SequencedStructuredModel([
            LengthFinishReasonError(
                "Could not parse response content as the length limit was reached"
            ),
            {"parsed": CONTRACT, "parsing_error": None},
        ])
        route = {
            "label": "REQUIRES_SCOPE_CONTRACT", "label_id": 1,
            "probability_requires_scope": 0.9, "threshold": 0.1,
            "router_status": "ok",
        }
        with patch.dict(os.environ, {"SCOPE_RESOLVER_MAX_ATTEMPTS": "3"}), patch(
            "scope_resolution.classify_scope_request", AsyncMock(return_value=route),
        ):
            updated, calls = await prepare_scope_context(context(), model)
        self.assertEqual(calls, 2)
        self.assertEqual(updated.scope_router["resolver_status"], "ok")
        self.assertIn("缩短思考过程", model.message_history[1][-1]["content"])

    async def test_invalid_schema_stops_after_three_attempts(self):
        invalid = {
            "raw": {"content": "{}"},
            "parsed": None,
            "parsing_error": ValueError("invalid contract"),
        }
        model = SequencedStructuredModel([invalid, invalid, invalid])
        route = {
            "label": "REQUIRES_SCOPE_CONTRACT", "label_id": 1,
            "probability_requires_scope": 0.9, "threshold": 0.1,
            "router_status": "ok",
        }
        with patch.dict(os.environ, {"SCOPE_RESOLVER_MAX_ATTEMPTS": "3"}), patch(
            "scope_resolution.classify_scope_request", AsyncMock(return_value=route),
        ):
            updated, calls = await prepare_scope_context(context(), model)
        self.assertEqual(calls, 3)
        self.assertEqual(model.calls, 3)
        self.assertEqual(updated.scope_router["resolver_status"], "failed")
        self.assertEqual(updated.scope_router["resolver_attempts"], 3)

    def test_scheduler_receives_contract_after_original_request(self):
        ctx = context().model_copy(update={
            "scope_router": {"label": "REQUIRES_SCOPE_CONTRACT"},
            "scope_contract": CONTRACT,
        })
        session = SchedulerConversation({}, ctx)
        session.initialize()
        kinds = [record["kind"] for record in session.records]
        self.assertIn("当前任务", kinds)
        self.assertIn("已校验范围合同", kinds)
        self.assertLess(kinds.index("当前任务"), kinds.index("已校验范围合同"))
        contract_record = next(
            record["content"] for record in session.records
            if record["kind"] == "已校验范围合同"
        )
        self.assertIn("INTERSECTION", contract_record)
        self.assertIn("不要扩大、缩小", contract_record)


if __name__ == "__main__":
    unittest.main()
