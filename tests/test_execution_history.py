import tempfile
import unittest
import json
from pathlib import Path
from langchain_core.messages import AIMessage, ToolMessage
from workers.history_archive import history_scope,capture,catalog,read_execution_history,ExecutionHistoryMiddleware

class HistoryTests(unittest.TestCase):
    def test_default_reads_tail_and_omits_submission(self):
        with tempfile.TemporaryDirectory() as root, history_scope('a','general',root):
            capture([AIMessage(content='',tool_calls=[{'name':'lookup','args':{},'id':'c'}]),
                     ToolMessage(content='OLDER',tool_call_id='c'),
                     AIMessage(content='',tool_calls=[{'name':'write','args':{},'id':'d'}]),
                     ToolMessage(content='LATEST',tool_call_id='d'),
                     AIMessage(content='',tool_calls=[{'name':'report_general_result','args':{'summary':'REPORT_ONLY'},'id':'e'}]),
                     ToolMessage(content='REPORT_ONLY',tool_call_id='e')])
            page=read_execution_history.invoke({'page_chars':400})
            self.assertIn('LATEST',page['content'])
            full=read_execution_history.invoke({})
            self.assertNotIn('REPORT_ONLY',full['content'])
            self.assertNotIn('report_general_result',full['content'])

    def test_cross_worker_stable_reference_pagination_and_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            with history_scope('task-a','general-1',root):
                messages=[AIMessage(content='',tool_calls=[{'name':'lookup','args':{'id':7},'id':'call-1'}]),
                          ToolMessage(content='x'*30000,tool_call_id='call-1',status='error')]
                capture(messages); before=catalog(); capture(messages)
                self.assertEqual(before,catalog())
                ref=before[-1]['reference']
            with history_scope('task-a','code-2',root):
                self.assertEqual(catalog(),before)
                page=read_execution_history.invoke({'reference':ref})
                self.assertEqual(page['next_offset'],24000)
                tail=read_execution_history.invoke({'reference':ref,'offset':24000})
                combined=json.loads(tail['content']+page['content'])
                self.assertEqual(combined['message']['status'],'error')
                self.assertIsNotNone(ExecutionHistoryMiddleware().before_agent({},None))
                with self.assertRaises(ValueError):read_execution_history.invoke({'reference':'../secret'})
            with history_scope('task-b','web-1',root):
                self.assertEqual(read_execution_history.invoke({'reference':ref})['status'],'NOT_FOUND')

    def test_mutation_rejected_and_provider_metadata_private(self):
        with tempfile.TemporaryDirectory() as root, history_scope('a','w',root):
            capture([AIMessage(content='observed result',tool_calls=[{'name':'lookup','args':{},'id':'c'}],additional_kwargs={'reasoning_content':'private reasoning'})])
            ref=catalog()[0]['reference']
            self.assertNotIn('private reasoning',read_execution_history.invoke({'reference':ref})['content'])
            path=next(Path(root).rglob(ref+'.json')); path.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):read_execution_history.invoke({'reference':ref})
