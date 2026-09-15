import asyncio
import json
from types import SimpleNamespace as NS
from knowledge_rag.expansion import grants,read_page
from knowledge_rag.runtime import RetrievalHub,retrieval_scope,read_knowledge
from knowledge_rag.middleware import KnowledgeMiddleware
from test_knowledge_rag import RagTests


class ExpansionTests(RagTests):
    def test_path_keeps_first_and_last_three(self):
        from knowledge_rag.structured import display_path
        kb=self.open()
        try:
            parts=[str(i) for i in range(10)]
            self.assertEqual(display_path(parts,kb),'0 / 1 / 2 / … / 7 / 8 / 9')
        finally:kb.close()

    def test_general_graph_receives_reference_and_reads_parent(self):
        from unittest.mock import AsyncMock,patch
        from langchain_core.messages import AIMessage,ToolMessage
        from test_code_agents import ToolRecordingModel
        from workers.general_worker import create_general_worker
        kb=self.open();path=self.base/'guide.json'
        path.write_text('{"name":"SUM","description":"Adds numbers"}',encoding='utf-8')
        try:
            kb.ingest(path);hits=kb.search('SUM',scorer=lambda q,docs:[NS(index=i,score=.9) for i in range(len(docs))])
            hub=RetrievalHub(None,None,self.config);hub.rag=AsyncMock(return_value=hits)
            hub._run=lambda scope,operation:operation(kb)
            model=ToolRecordingModel(responses=[AIMessage(content='',tool_calls=[{'name':'read_knowledge','args':{'reference_id':'fixture'},'id':'read'}]),AIMessage(content='',tool_calls=[{'name':'report_general_result','args':{'result':{'status':'COMPLETED','summary':'verified'}},'id':'done'}])])
            graph=create_general_worker(model,tools=[])
            async def run():
                with retrieval_scope(hub,'owner','graph'),patch('knowledge_rag.expansion.uuid4',return_value=NS(hex='fixture')):
                    assignment='当前Step：\n'+json.dumps({'objective':'use SUM','rag_query':'SUM'})
                    return await graph.ainvoke({'messages':[{'role':'user','content':assignment}],'skill_mode':'off'},config={'configurable':{'thread_id':'general'}})
            result=asyncio.run(run())
            reads=[json.loads(m.content) for m in result['messages'] if isinstance(m,ToolMessage) and m.name=='read_knowledge']
            self.assertEqual(reads[0]['status'],'ok');self.assertIn('Adds numbers',reads[0]['text'])
            self.assertIn('read_knowledge',model.bound_tool_names)
        finally:kb.close()

    def test_pages_reassemble_and_version_change_rejected(self):
        kb=self.open();path=self.base/'big.json'
        original={'name':'example','description':'Many lines of useful information.\n'*300}
        path.write_text(json.dumps(original),encoding='utf-8')
        try:
            kb.ingest(path)
            hits=kb.search('example')
            grant=hits[0]
            parts=[];offset=0
            while True:
                page=read_page(kb,grant,offset)
                self.assertEqual(page['status'],'ok');self.assertLessEqual(page['tokens'],2000)
                parts.append(page['text']);offset=page['next_offset']
                if offset is None:break
            self.assertEqual(json.loads(''.join(parts)),original)
            path.write_text('{"name":"new"}',encoding='utf-8');kb.ingest(path)
            self.assertEqual(read_page(kb,grant,0)['status'],'stale_reference')
        finally:kb.close()

    def test_tool_visibility_requires_agent_scoped_qualified_hit(self):
        hub=RetrievalHub(None,None,self.config)
        tools=[NS(name='search_knowledge'),NS(name='read_knowledge'),NS(name='appworld_execute')]
        class Request:
            state={'knowledge_agent_key':'a'}
            def __init__(self):self.tools=tools
            def override(self,**kw):return NS(**kw)
        middleware=KnowledgeMiddleware('General')
        self.assertEqual([t.name for t in middleware.filtered(Request()).tools],['appworld_execute'])
        with retrieval_scope(hub,'owner','run'):
            self.assertEqual(len(middleware.filtered(Request()).tools),1)
            hit={'relevance_score':.39,'_source_id':'s','_version':'v','_collection':'c','parent_id':'p'}
            self.assertEqual(grants(hub,'run','a',[hit]),[])
            grants(hub,'run','other',[hit|{'relevance_score':.9}])
            self.assertEqual(len(middleware.filtered(Request()).tools),1)
            grants(hub,'run','a',[hit|{'relevance_score':.9}])
            self.assertEqual([t.name for t in middleware.filtered(Request()).tools],['search_knowledge','read_knowledge','appworld_execute'])
            request=Request();request.tools=[{'name':t.name} for t in tools]
            self.assertEqual(len(middleware.filtered(request).tools),3)
            request.state={'knowledge_agent_key':'unrelated'}
            self.assertEqual(middleware.filtered(request).tools,[{'name':'appworld_execute'}])

    def test_all_worker_roles_auto_retrieve_once(self):
        from unittest.mock import AsyncMock
        hub=RetrievalHub(None,None,self.config);hub.rag=AsyncMock(return_value=[])
        async def run():
            with retrieval_scope(hub,'owner','run'):
                for role in ['General Agent','Code Worker','Code Reviewer','Web Agent']:
                    m=KnowledgeMiddleware(role)
                    state={'messages':[{'role':'user','content':'当前Step：\n'+json.dumps({'rag_query':'task'})}]}
                    update=await m.abefore_agent(state,None,{'configurable':{'thread_id':'shared'}})
                    self.assertIsNone(await m.abefore_agent(state|update,None,{'configurable':{'thread_id':'shared'}}))
        asyncio.run(run())
        self.assertEqual(hub.rag.await_count,4)
