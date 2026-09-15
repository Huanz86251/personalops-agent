import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from test_knowledge_rag import RagTests
from knowledge_rag.runtime import RetrievalHub,retrieval_scope,SHARED_CODE
from knowledge_rag.middleware import KnowledgeMiddleware
from knowledge_rag.expansion import grants,read_page
from knowledge_rag.query import task_query

class SharedTests(RagTests):
    def test_query_uses_step_and_contract_not_budget(self):
        assignment='只执行当前Step；不要生成StepReport。\n用户原始请求：Spotify\n整体目标：rating\n当前Step（第1次尝试）：\n'+json.dumps({'objective':'verify ratings','success_criteria':['preserve comments']})+'\n本次Attempt预算：15工具'
        q=task_query(assignment)
        self.assertIn('verify ratings',q)
        self.assertNotIn('预算',q)
        self.assertNotIn('StepReport',q)
        q=task_query('four rounds remaining',{'requirements':[{'statement':'Spotify ratings'}],'implementation_guidance':['submit JSON']})
        self.assertIn('Spotify',q);self.assertNotIn('four rounds',q);self.assertNotIn('submit JSON',q)

    def test_code_boundary_shares_one_retrieval_and_readable_grants(self):
        from agent import ask_worker
        from langchain_core.messages import AIMessage
        kb=self.open();doc=self.base/'apis.json'
        doc.write_text('[{"api_name":"login","description":"Authenticate"},{"api_name":"ratings","description":"Check ratings"}]')
        try:
            kb.ingest(doc);hits=kb.search('ratings',scorer=lambda q,docs:[NS(index=i,score=.9) for i in range(len(docs))])
            hub=RetrievalHub(None,None,self.config);hub.rag=AsyncMock(return_value=hits)
            seen=[]
            async def invoke(state,config):
                self.assertIsNotNone(SHARED_CODE.get())
                self.assertNotIn('文档检索资料',state.get('execution_instructions',''))
                for role in ['Code Worker','Code Reviewer']:
                    captured=[]
                    async def inner(inner_state,config):
                        captured.append(await KnowledgeMiddleware(role).abefore_agent(inner_state,None,config))
                        return {'messages':[AIMessage(content='done')]}
                    await ask_worker(NS(ainvoke=inner),'budget only',thread_id='code:'+role,trace_role='code' if role=='Code Worker' else 'reviewer')
                    update=captured[0]
                    refs=grants(hub,'run',update['knowledge_agent_key'])
                    self.assertTrue(refs)
                    self.assertEqual(read_page(kb,next(iter(refs.values())),0)['status'],'ok')
                    seen.append([v['node_id'] for v in refs.values()])
                return {'messages':[AIMessage(content='done')]}
            async def run():
                with retrieval_scope(hub,'owner','run'):
                    await ask_worker(NS(ainvoke=invoke),'当前Step：\n'+json.dumps({'rag_query':'Spotify ratings'}),thread_id='code',trace_role='code_agent',state_update={'code_task':{'requirements':[{'statement':'ratings'}]}})
                self.assertIsNone(SHARED_CODE.get())
            asyncio.run(run());self.assertEqual(hub.rag.await_count,1);self.assertEqual(seen[0],seen[1])
            outline=read_page(kb,hits[0],0,view='children',node_id=hits[0]['root_node_id'],depth=1)
            self.assertTrue(any('[0]' in e['title'] and 'login' in e['title'] for e in outline['entries']))
        finally:kb.close()

    def test_shared_empty_does_not_requery(self):
        from knowledge_rag.runtime import shared_code_scope
        hub=RetrievalHub(None,None,self.config);hub.rag=AsyncMock()
        async def run():
            with retrieval_scope(hub,'owner','empty'),shared_code_scope('empty-owner'):
                update=await KnowledgeMiddleware('Code Reviewer').abefore_agent({'messages':[]},None,{'configurable':{'thread_id':'reviewer'}})
                self.assertEqual(update['knowledge_context'],'')
                hub.rag.assert_not_awaited()
        asyncio.run(run())

    def test_nested_scope_isolation_and_exception_restore(self):
        from agent import ask_worker
        from langchain_core.messages import AIMessage
        hub=RetrievalHub(None,None,self.config);hub.rag=AsyncMock(return_value=[])
        async def outer(state,config):
            expected='code-task:'+config['configurable']['thread_id']
            async def inner(state,config):
                await asyncio.sleep(.01)
                self.assertEqual(SHARED_CODE.get(),expected)
                update=await KnowledgeMiddleware('Code Worker').abefore_agent(state,None,config)
                self.assertEqual(update['knowledge_context'],'')
                return {'messages':[AIMessage(content='done')]}
            await ask_worker(NS(ainvoke=inner),'budget',thread_id=expected+':worker',trace_role='code')
            async def failing(state,config):
                self.assertIsNone(SHARED_CODE.get())
                raise RuntimeError('fixture')
            with self.assertRaises(RuntimeError):
                await ask_worker(NS(ainvoke=failing),'independent',thread_id=expected+':general',trace_role='general')
            self.assertEqual(SHARED_CODE.get(),expected)
            return {'messages':[AIMessage(content='done')]}
        async def run():
            with retrieval_scope(hub,'owner','isolated'):
                await asyncio.gather(*(ask_worker(NS(ainvoke=outer),'当前Step：\n'+json.dumps({'rag_query':'query records'}),thread_id=k,trace_role='code_agent') for k in ['a','b']))
            self.assertIsNone(SHARED_CODE.get())
        asyncio.run(run());self.assertEqual(hub.rag.await_count,2)
