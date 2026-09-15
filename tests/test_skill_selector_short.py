import unittest,asyncio,json
from unittest.mock import patch
from skill_runtime.preparation import _messages,skill_selector_scope,prepare_skills,SELECTOR_MODEL
from test_skill_preparation import ProbeModel

class ShortTests(unittest.TestCase):
    def test_no_system_no_execution_payload(self):
        messages=_messages('code',{'user_request':'Spotify task','execution_instructions':'SECRET_SYSTEM','memory_context':'SECRET_MEMORY','step_reports':['SECRET_REPORT']},[])
        self.assertEqual([m['role'] for m in messages],['user'])
        self.assertIn('Spotify',messages[0]['content'])
        self.assertNotIn('SECRET',messages[0]['content'])
    def test_all_roles_use_shared_selector_and_restore(self):
        executor=ProbeModel();selector=ProbeModel(selected=[])
        async def run():
            with skill_selector_scope(selector):
                for role in ['scheduler','general','code','reviewer','web','step_reporter']:
                    await prepare_skills(executor,role=role,task='task',mode='dynamic',tools=['execute','read_file','web_search','web_fetch'])
            self.assertIsNone(SELECTOR_MODEL.get())
        asyncio.run(run())
        self.assertEqual(len(selector.selection_requests),6)
        self.assertEqual(executor.selection_requests,[])
        self.assertTrue(all(len(m)==1 and m[0]['role']=='user' for m in selector.selection_requests))
    def test_config_independent(self):
        import os
        from test_model_roles import defaults
        from model_roles import load_role_models
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'fixture','OPENAI_API_KEY':'fixture','SKILL_SELECTOR_LLM_MODEL':'cheap-selector'},clear=True):
            roles=load_role_models(defaults())
        self.assertEqual(roles['skill_selector'].model,'cheap-selector')
        self.assertEqual(roles['code'].model,'deepseek-v4-pro')
        self.assertFalse(roles['skill_selector'].thinking_enabled)
        self.assertEqual(roles['skill_selector'].max_tokens,512)

class ReviewerShortTests(unittest.TestCase):
    def test_review_contract_names_survive_but_tool_results_do_not(self):
        task={'task_contract': {'user_request':'AppWorld notes task','step_assignment':'Verify saved notes',
                               'success_criteria':['Preserve original body'],'execution_guidance':'SECRET_POLICY'},
              'recent_attempt_outcomes':[{'has_errors':True,'tool_result_excerpts':['SECRET_OUTPUT']}]}
        msg=_messages('step_reporter',task,[])
        self.assertEqual([m['role'] for m in msg],['user'])
        self.assertIn('AppWorld',msg[0]['content'])
        self.assertIn('Verify saved notes',msg[0]['content'])
        self.assertNotIn('SECRET',msg[0]['content'])

    def test_appworld_reviewer_discovery_and_selected_body(self):
        from skill_runtime import prepare_skills_sync, skill_prompt
        model=ProbeModel(selected=['review-appworld-results'])
        snap=prepare_skills_sync(model,role='step_reporter',task='AppWorld task review',topics=['appworld'],mode='dynamic')
        self.assertEqual([s.name for s in snap.selected],['review-appworld-results'])
        self.assertIn('原文',skill_prompt(snap))
        self.assertNotIn('read_review_material(reference=',model.selection_requests[0][0]['content'])
