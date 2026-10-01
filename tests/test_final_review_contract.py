"""Offline replay of invalid Final Review branches and preserved task inputs."""
import json
import unittest
from pathlib import Path
from pydantic import ValidationError
from hard_planning import _invoke_structured, run_hard_final_reviewer
from planning_models import FinalReviewDecision, PlanningContextPack, StepReport
from planning_graph import _append_verified_source_links
from step_execution import _reporter_source_links
from reporting.models import ReviewAttempt, ReviewTaskContract, StepReviewPacket
from workers.submission import ResolvedToolEvidence
from datetime import datetime, timezone
from test_planning_repair_prefix import Scripted


class ReviewContractTests(unittest.IsolatedAsyncioTestCase):
    def test_harness_extracts_cited_browser_navigation_source(self):
        packet = StepReviewPacket(
            created_at=datetime.now(timezone.utc),
            task_contract=ReviewTaskContract(
                user_request="推荐健身房",
                plan_objective="推荐健身房",
                step_id=1,
                step_assignment="查公开资料",
                success_criteria=["给出来源"],
            ),
            attempts=[ReviewAttempt(
                attempt=1,
                submission_source="WORKER",
                finish_reason="submitted",
                stop_reason="done",
                summary="done",
                final_conclusion="done",
                resolved_evidence=[ResolvedToolEvidence(
                    tool_call_id="browser-call",
                    tool_name="browser_navigate",
                    arguments={"url": "https://example.com/laixi-gym"},
                    result=(
                        "Page URL: https://example.com/laixi-gym/final\n"
                        "Page title: 莱西健身房介绍"
                    ),
                    result_chars=80,
                )],
            )],
        )
        links = _reporter_source_links(
            packet,
            {"browser-call": "E1"},
            {"browser-call"},
        )
        self.assertEqual(links, [{
            "evidence_ref": "browser-call",
            "title": "莱西健身房介绍",
            "href": "https://example.com/laixi-gym/final",
        }])

    def test_harness_appends_only_final_reviewer_cited_web_links(self):
        report = StepReport(
            step_id=1,
            status="COMPLETED",
            summary="已核对候选。",
            stop_reason="done",
            source_links=[
                {"evidence_ref": "WA1", "title": "莱西餐厅页面", "href": "https://example.com/a"},
                {"evidence_ref": "WB1", "title": "未采用页面", "href": "https://example.com/b"},
            ],
        )
        decision = FinalReviewDecision(
            action="FINAL",
            status="COMPLETED",
            final_answer="已完成推荐。",
            criterion_reviews=[{
                "criterion_id": "C1",
                "evidence_refs": ["WA1"],
                "observed_result": "已核对。",
                "missing_requirement": None,
                "status": "MET",
            }],
        )

        answer = _append_verified_source_links(
            decision.final_answer or "", decision, [report]
        )

        self.assertIn("🌐 **莱西餐厅页面**", answer)
        self.assertIn("[example.com](https://example.com/a)", answer)
        self.assertIn("`WA1`", answer)
        self.assertNotIn("https://example.com/b", answer)

    def test_harness_keeps_cited_official_fetch_even_when_link_is_already_inline(self):
        packet = StepReviewPacket(
            created_at=datetime.now(timezone.utc),
            task_contract=ReviewTaskContract(
                user_request="查询故宫开放时间",
                plan_objective="查询故宫开放时间",
                step_id=1,
                step_assignment="查官方资料",
                success_criteria=["给出官方来源"],
            ),
            attempts=[ReviewAttempt(
                attempt=1,
                submission_source="WORKER",
                finish_reason="submitted",
                stop_reason="done",
                summary="done",
                final_conclusion="done",
                resolved_evidence=[ResolvedToolEvidence(
                    tool_call_id="official-fetch",
                    tool_name="fetch_webpage",
                    arguments={"url": "https://www.dpm.org.cn/visit.html"},
                    result=json.dumps({
                        "fetch_status": "SUCCESS",
                        "final_url": "https://www.dpm.org.cn/visit.html",
                        "source_url": "https://www.dpm.org.cn/visit.html",
                        "content": "# 参观须知\n开放时间如下",
                    }, ensure_ascii=False),
                    result_chars=160,
                )],
            )],
        )
        links = _reporter_source_links(
            packet, {"official-fetch": "official-fetch"}, {"official-fetch"},
        )
        report = StepReport(
            step_id=1,
            status="COMPLETED",
            summary="已核对官方页面。",
            stop_reason="done",
            source_links=links,
        )
        decision = FinalReviewDecision(
            action="FINAL",
            status="COMPLETED",
            final_answer="官网说明见 https://www.dpm.org.cn/visit.html 。",
            criterion_reviews=[{
                "criterion_id": "C1",
                "evidence_refs": ["official-fetch"],
                "observed_result": "已核对官网。",
                "missing_requirement": None,
                "status": "MET",
            }],
        )
        answer = _append_verified_source_links(decision.final_answer or "", decision, [report])
        self.assertIn("🌐 **参观须知**", answer)
        self.assertIn("[www.dpm.org.cn](https://www.dpm.org.cn/visit.html)", answer)

    def test_completed_final_review_inherits_verified_step_sources(self):
        report = StepReport(
            step_id=1,
            status="COMPLETED",
            summary="已核对官方页面。",
            stop_reason="done",
            criterion_results=[{
                "criterion_id": "C1",
                "criterion": "给出官方开放时间",
                "evidence": ["WA1", "E2"],
                "status": "MET",
            }],
            source_links=[
                {
                    "evidence_ref": "E2",
                    "title": "故宫博物院导览",
                    "href": "https://www.dpm.org.cn/Visit.html",
                },
                {
                    "evidence_ref": "WA1",
                    "title": "搜索结果标题",
                    "href": "https://www.dpm.org.cn/Visit.html",
                },
            ],
        )
        decision = FinalReviewDecision(
            action="FINAL",
            status="COMPLETED",
            final_answer="故宫明日正常开放。",
            criterion_reviews=[{
                "criterion_id": "C1",
                "evidence_refs": [],
                "observed_result": "官方页面已核对。",
                "missing_requirement": None,
                "status": "MET",
            }],
        )

        answer = _append_verified_source_links(
            decision.final_answer or "", decision, [report]
        )

        self.assertIn("故宫博物院导览", answer)
        self.assertIn("https://www.dpm.org.cn/Visit.html", answer)
        self.assertNotIn("搜索结果标题", answer)
        self.assertEqual(answer.count("https://www.dpm.org.cn/Visit.html"), 1)

    async def test_budget_exhaustion_produces_explicit_harness_report(self):
        class Exhausted(Scripted):
            def __init__(self):
                super().__init__([])

            def with_structured_output(self, *args, **kwargs):
                class Bound:
                    async def ainvoke(self, messages):
                        raise RuntimeError("Trial model-call budget exhausted")
                return Bound()

        result = await run_hard_final_reviewer(
            Exhausted(),
            context=PlanningContextPack(current_time="now", user_request="finish safely"),
            plan_objective="finish safely",
            plan_success_criteria=["verified"],
            step_reports=[],
            replan_history=[],
            overall_stop_reason="budget",
            replan_available=False,
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.output.status, "FAILED")
        self.assertIn("预算已耗尽", result.output.final_answer)
        self.assertIn("不是 Final Reviewer", result.output.final_answer)

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

    async def test_final_reviewer_does_not_repeat_a_length_exhausted_call(self):
        class LengthFinishReasonError(RuntimeError):
            pass

        class ExhaustedOnce(Scripted):
            def __init__(self):
                super().__init__([])
                self.calls = 0

            def with_structured_output(self, *args, **kwargs):
                owner = self
                class Bound:
                    async def ainvoke(self, messages):
                        owner.calls += 1
                        raise LengthFinishReasonError("finish_reason=length")
                return Bound()

        model = ExhaustedOnce()
        result = await run_hard_final_reviewer(
            model,
            context=PlanningContextPack(current_time="now", user_request="finish safely"),
            plan_objective="finish safely",
            plan_success_criteria=["verified"],
            step_reports=[],
            replan_history=[],
            overall_stop_reason="length",
            replan_available=False,
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(model.calls, 1)

    async def test_final_review_projects_reports_without_raw_duplicate_fields(self):
        model = Scripted([{'parsed': {
            'action': 'FINAL', 'status': 'COMPLETED', 'final_answer': '已核对。',
            'unmet_success_criteria': [], 'replan_reason': None,
        }}])
        from planning_models import StepReport
        report = StepReport(
            step_id=1,
            status="COMPLETED",
            summary="已核对。",
            answer_text="完整交付正文",
            stop_reason="done",
            completed_work=["RAW_DUPLICATE_WORK"],
            worker_contributions=[{"worker_id": "w", "contribution": "RAW_DUPLICATE_CONTRIBUTION"}],
        )
        await run_hard_final_reviewer(
            model,
            context=PlanningContextPack(current_time="now", user_request="核对"),
            plan_objective="核对",
            plan_success_criteria=["已核对"],
            step_reports=[report],
            replan_history=[],
            overall_stop_reason="done",
            replan_available=False,
        )
        wire = json.dumps(model.requests[0], ensure_ascii=False)
        self.assertIn("完整交付正文", wire)
        self.assertNotIn("RAW_DUPLICATE_WORK", wire)
        self.assertNotIn("RAW_DUPLICATE_CONTRIBUTION", wire)

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
            completion_api_contract='PUBLIC-COMPLETE-TASK-DOC',
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
        self.assertIn('SHOULD_NOT_APPEAR', wire)
        self.assertIn('PUBLIC-COMPLETE-TASK-DOC', wire)

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

    def test_stringified_repair_request_is_decoded_by_harness(self):
        repair = {
            "step_id": 1,
            "worker_kind": "GENERAL",
            "failed_criterion_ids": ["C2"],
            "evidence_refs": [],
            "observed_problem": "没有观察到发送回执。",
            "missing_requirement": "需要真实发送成功回执。",
        }
        decision = FinalReviewDecision.model_validate({
            "review_reason": "C2缺少完成回执。",
            "criterion_reviews": [{
                "criterion_id": "C2",
                "evidence_refs": [],
                "observed_result": "未观察到发送回执。",
                "missing_requirement": "缺少成功回执。",
                "status": "NOT_MET",
            }],
            "repair_request": json.dumps(repair, ensure_ascii=False),
            "replan_reason": None,
            "action": "RETURN_TO_WORKER",
            "status": None,
            "final_answer": None,
            "unmet_success_criteria": [],
        })
        self.assertEqual(decision.repair_request.step_id, 1)
        self.assertEqual(decision.repair_request.failed_criterion_ids, ["C2"])

        with self.assertRaises(ValidationError):
            FinalReviewDecision.model_validate({
                "review_reason": "坏的嵌套对象仍需拒绝。",
                "criterion_reviews": [],
                "repair_request": "not json",
                "replan_reason": None,
                "action": "RETURN_TO_WORKER",
                "status": None,
                "final_answer": None,
                "unmet_success_criteria": [],
            })

    def test_prompt_explicitly_requests_json_for_compatible_providers(self):
        text = Path('prompts/planning/final_reviewer.md').read_text(encoding='utf-8')
        self.assertIn('JSON', text.upper())

    def test_prompt_uses_small_action_examples_instead_of_schema_copies(self):
        text=Path('prompts/planning/final_reviewer.md').read_text(encoding='utf-8')
        self.assertIn('`RETURN_TO_WORKER`', text)
        self.assertIn('`FINAL`', text)
        self.assertIn('`REPLAN`', text)
        self.assertNotIn('FINAL：{"review_reason"', text)
        self.assertNotIn('REPLAN：{"review_reason"', text)
        with self.assertRaises(ValidationError):
            FinalReviewDecision.model_validate({'action':'REPLAN','status':'PARTIAL','replan_reason':'continue'})
