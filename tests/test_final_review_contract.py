"""Offline replay of invalid Final Review branches and preserved task inputs."""
import json
import unittest
from pathlib import Path
from pydantic import ValidationError
from hard_planning import _invoke_structured, run_hard_final_reviewer
from planning_models import FinalReviewDecision, PlanningContextPack
from test_planning_repair_prefix import Scripted


class ReviewContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_observed_invalid_branch_can_repair_without_changing_prefix(self):
        invalid = {'action': 'REPLAN', 'unmet_success_criteria': ['unfinished'], 'replan_reason': 'continue'}
        valid = {'action': 'REPLAN', 'status': None, 'final_answer': None,
                 'unmet_success_criteria': [], 'replan_reason': 'Inputs obtained; execute then verify'}
        model = Scripted([{'parsed': invalid}, {'parsed': valid}])
        result = await _invoke_structured(model, prompt='fixed', output_schema=FinalReviewDecision,
            trace_name='offline-review', scheduler_messages=[{'role': 'user', 'content': 'original scope'}],
            fallback_factory=lambda error: FinalReviewDecision(action='FINAL', status='FAILED', final_answer='failed'))
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.output.action, 'REPLAN')
        self.assertEqual(model.requests[1][:len(model.requests[0])], model.requests[0])
        self.assertIn('unmet_success_criteria=[]', model.requests[1][-1]['content'])

    async def test_initial_plus_three_invalid_repairs_stop(self):
        model = Scripted([
            {'parsed': {'action': 'PLAN'}},
            {'parsed': {'action': 'REPLAN', 'unmet_success_criteria': ['unfinished'], 'replan_reason': 'continue'}},
            {'parsed': {'action': 'REPLAN', 'unmet_success_criteria': ['unfinished'], 'replan_reason': 'continue'}},
            {'parsed': {'action': 'REPLAN', 'unmet_success_criteria': ['unfinished'], 'replan_reason': 'continue'}},
        ])
        result = await _invoke_structured(model, prompt='fixed', output_schema=FinalReviewDecision,
            trace_name='offline-review', fallback_factory=lambda error:
            FinalReviewDecision(action='FINAL', status='FAILED', final_answer='failed'))
        self.assertTrue(result.used_fallback)
        self.assertEqual(len(model.requests), 4)

    async def test_original_conditions_reach_final_review_despite_lossy_plan(self):
        for request in ('只更新本月已批准的订单，保留取消订单。', 'Archive unread messages in folder Work, except flagged messages.'):
            model = Scripted([{'parsed': {'action': 'FINAL', 'status': 'FAILED', 'final_answer': 'scope unverified'}}])
            await run_hard_final_reviewer(model, context=PlanningContextPack(current_time='now', user_request=request),
                plan_objective='process all records', plan_success_criteria=['processed'], step_reports=[],
                replan_history=[], overall_stop_reason='blocked', replan_available=True)
            wire=json.dumps(model.requests[0], ensure_ascii=False)
            self.assertIn(request, wire)
            self.assertIn('计划自身不能替代原请求', wire)

    async def test_final_review_uses_independent_minimal_evidence_pack(self):
        context = PlanningContextPack(
            current_time='SHOULD_NOT_APPEAR',
            user_request='核对全部订单',
            toolset_catalog=[{'name': 'CAPABILITY_SHOULD_NOT_APPEAR'}],
        )
        context.scheduler_session['records'] = [{
            'role': 'user', 'content': 'OLD_PLAN_PROTOCOL_SHOULD_NOT_APPEAR',
            'kind': 'protocol', 'protected': False, 'key': 'supervisor@old',
        }]
        model = Scripted([{'parsed': {
            'action': 'FINAL', 'status': 'COMPLETED', 'final_answer': '已核对。',
            'unmet_success_criteria': [], 'replan_reason': None,
        }}])
        await run_hard_final_reviewer(
            model, context=context, plan_objective='核对订单',
            plan_success_criteria=['全部已核对'], step_reports=[],
            replan_history=[{'reason': '一次调整'}], overall_stop_reason='done',
            replan_available=False,
        )
        wire = json.dumps(model.requests[0], ensure_ascii=False)
        self.assertIn('核对全部订单', wire)
        self.assertIn('一次调整', wire)
        self.assertNotIn('OLD_PLAN_PROTOCOL_SHOULD_NOT_APPEAR', wire)
        self.assertNotIn('CAPABILITY_SHOULD_NOT_APPEAR', wire)
        self.assertNotIn('SHOULD_NOT_APPEAR', wire)

    def test_return_to_worker_is_evidence_first_and_has_no_api_advice(self):
        decision = FinalReviewDecision.model_validate({
            "review_reason": "C2缺少完成回执。",
            "criterion_reviews": [{
                "criterion_id": "C2",
                "evidence_refs": [],
                "observed_result": "未观察到完成提交。",
                "missing_requirement": "缺少成功回执。",
                "status": "NOT_MET",
            }],
            "repair_request": {
                "step_id": 1,
                "worker_kind": "GENERAL",
                "failed_criterion_ids": ["C2"],
                "evidence_refs": [],
                "observed_problem": "未观察到完成提交。",
                "missing_requirement": "缺少成功回执。",
            },
            "replan_reason": None,
            "action": "RETURN_TO_WORKER",
            "status": None,
            "final_answer": None,
            "unmet_success_criteria": [],
        })
        self.assertEqual(decision.action, "RETURN_TO_WORKER")
        fields = list(FinalReviewDecision.model_fields)
        self.assertLess(fields.index("criterion_reviews"), fields.index("action"))
        self.assertLess(fields.index("repair_request"), fields.index("action"))

    def test_prompt_explicitly_requests_json_for_compatible_providers(self):
        text = Path('prompts/planning/final_reviewer.md').read_text(encoding='utf-8')
        self.assertIn('JSON', text.upper())

    def test_documented_examples_follow_actual_contract(self):
        text=Path('prompts/planning/final_reviewer.md').read_text(encoding='utf-8')
        examples=[json.loads(line.split('：',1)[1]) for line in text.splitlines() if line.startswith(('FINAL：','REPLAN：'))]
        self.assertEqual(len(examples), 2)
        for example in examples:
            FinalReviewDecision.model_validate(example)
        with self.assertRaises(ValidationError):
            FinalReviewDecision.model_validate({'action':'REPLAN','status':'PARTIAL','replan_reason':'continue'})
