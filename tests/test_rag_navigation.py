import asyncio
import json
from types import SimpleNamespace as NS
from test_knowledge_rag import RagTests
from knowledge_rag.expansion import read_page,grants
from knowledge_rag.runtime import RetrievalHub,retrieval_scope,read_knowledge


class NavigationTests(RagTests):
    def test_directory_then_sibling_read_and_parent_expansion(self):
        kb=self.open();p=self.base/'paper.json'
        p.write_text(json.dumps({'Introduction':{'text':'Fish farming introduction'},'Methods':{'Terms':{'text':'Precise terminology'}}}),encoding='utf-8')
        try:
            kb.ingest(p);hit=kb.search('Fish')[0]
            outline=read_page(kb,hit,0,view='children',node_id=hit['root_node_id'],depth=1)
            method=next(e for e in outline['entries'] if e['title']=='Methods')
            sub=read_page(kb,hit,0,view='children',node_id=method['node_id'],depth=1)
            term=next(e for e in sub['entries'] if e['title']=='Terms')
            page=read_page(kb,hit,0,node_id=term['node_id'])
            self.assertIn('Precise terminology',page['text'])
            self.assertEqual(read_page(kb,hit,0,node_id='other-document')['status'],'invalid_node')
        finally:kb.close()

    def test_outline_pages_are_bounded_complete_and_depth_limited(self):
        kb=self.open();p=self.base/'many.json'
        p.write_text(json.dumps({f'section{i}':{'child':i} for i in range(180)}),encoding='utf-8')
        try:
            kb.ingest(p);hit=kb.search('section0')[0];offset=0;ids=[];pages=0
            while True:
                result=read_page(kb,hit,offset,view='children',node_id=hit['root_node_id'],depth=1)
                self.assertLessEqual(result['tokens'],2000)
                self.assertTrue(all(e['depth']==1 for e in result['entries']))
                ids.extend(e['node_id'] for e in result['entries']);pages+=1
                offset=result['next_offset']
                if offset is None:break
            self.assertEqual(len(set(ids)),180);self.assertEqual(len(ids),180);self.assertGreater(pages,1)
        finally:kb.close()

    def test_read_tool_does_not_stop_after_two_calls(self):
        kb=self.open();p=self.base/'guide.json';p.write_text('{"topic":"facts"}',encoding='utf-8')
        try:
            kb.ingest(p);hit=kb.search('facts')[0]|{'relevance_score':.9}
            hub=RetrievalHub(None,None,self.config);hub._run=lambda scope,operation:operation(kb)
            ref=grants(hub,'run','agent',[hit])[0]['reference_id']
            async def run():
                with retrieval_scope(hub,'owner','run'):
                    return [await read_knowledge.coroutine(ref,NS(state={'knowledge_agent_key':'agent'},config={}),view='children') for _ in range(4)]
            self.assertEqual([x['status'] for x in asyncio.run(run())],['ok']*4)
        finally:kb.close()
