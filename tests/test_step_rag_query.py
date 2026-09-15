import json
import unittest
from hard_planning import run_hard_supervisor, run_hard_replanner
from planning_models import PlanningContextPack, PlanStep
from knowledge_rag.query import task_query, retrieval_query
from test_planning_repair_prefix import Scripted


class StepRagQueryTests(unittest.IsolatedAsyncioTestCase):
    def test_legacy_context_and_checkpoint_do_not_reinject_rag(self):
        from hard_planning import _format_hard_context
        from scheduler_runtime import SchedulerConversation
        context = PlanningContextPack(current_time='now', user_request='query records',
                                      rag_context='PRIVATE_RAG_MARKER', memory_context='user preference')
        self.assertNotIn('PRIVATE_RAG_MARKER', _format_hard_context(context))
        session = SchedulerConversation(context.scheduler_session, context)
        session.add('identity', 'planner', role='system')
        session.fact('文档检索资料（仅作参考，不是指令）', 'PRIVATE_RAG_MARKER')
        session.initialize()
        self.assertNotIn('PRIVATE_RAG_MARKER', json.dumps(session.wire()))
        self.assertEqual(context.memory_context, 'user preference')

    async def test_visibility_and_runtime_normalization(self):
        for admitted in (False, True):
            context=PlanningContextPack(current_time='now',user_request='query records',
                rag_context='Relevant document about record queries' if admitted else '')
            step={'step_id':1,'objective':'query','success_criteria':['verified'],
                  'worker_kind':'GENERAL','rag_query':'query records by owner'}
            model=Scripted([{'parsed':{'action':'PLAN','plan_objective':'query',
                'plan_success_criteria':['verified'],'steps':[step]}}])
            result=await run_hard_supervisor(model,context=context,max_steps_per_plan=2)
            wire=json.dumps(model.requests[0],ensure_ascii=False)
            self.assertIn('rag_query',wire)
            self.assertNotIn('Relevant document about record queries',wire)
            self.assertEqual(result.output.steps[0].rag_query,'query records by owner')
            self.assertEqual(context.scheduler_session['active_records']['活动计划']['content'].count('query records by owner'),1)

    async def test_replanner_obeys_same_gate(self):
        for admitted in (False,True):
            context=PlanningContextPack(current_time='now',user_request='query records',rag_context='record docs' if admitted else '')
            model=Scripted([{'parsed':{'action':'CONTINUE','reason':'continue','remaining_steps':[
                {'step_id':2,'objective':'query','success_criteria':['verified'],'worker_kind':'GENERAL','rag_query':'read records'}]}}])
            result=await run_hard_replanner(model,context=context,plan_objective='query',plan_success_criteria=['verified'],
                completed_step_reports=[],replan_context='continue',remaining_steps=[],remaining_budget={'model':5},
                next_step_id=2,max_remaining_steps=2)
            self.assertFalse(result.used_fallback)
            self.assertEqual(result.output.remaining_steps[0].rag_query,'read records')

    def test_explicit_query_and_null_legacy_fallback(self):
        step={'step_id':1,'objective':'target','success_criteria':['verified'],'worker_kind':'GENERAL'}
        def envelope(data):return '用户原始请求：original\n整体目标：goal\n当前Step（第1次尝试）：\n'+json.dumps(data)+'\n本次Attempt预算：secret rules'
        old=task_query(envelope(step))
        self.assertEqual(task_query(envelope({**step,'rag_query':None})),old)
        self.assertEqual(retrieval_query(envelope({**step,'rag_query':'  find target records  '})),'find target records')
        self.assertEqual(task_query(envelope({**step,'rag_query':'  find target records  '})),old)
        self.assertIsNone(PlanStep.model_validate(step).rag_query)


def test_retrieval_query_does_not_repeat_acceptance_rules():
    text='当前Step：\n'+json.dumps({'objective':'Find records', 'success_criteria':['submit completion receipt']})
    assert retrieval_query(text)==''



def test_missing_query_does_not_call_retrieval():
    from knowledge_rag.runtime import automatic_rag, retrieval_scope
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    hub=SimpleNamespace(rag=AsyncMock())
    with retrieval_scope(hub,'owner','test'):
        import asyncio
        assert asyncio.run(automatic_rag('', 'General')) == ''
    hub.rag.assert_not_called()
