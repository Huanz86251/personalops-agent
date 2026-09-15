"""Free integration tests; models and outbound delivery are replaced locally."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from knowledge_rag.runtime import RetrievalHub, retrieval_scope, automatic_rag, search_knowledge, read_knowledge
from knowledge_rag.ingress import RagUploadInbox
from knowledge_rag.service import PROJECT
from test_knowledge_rag import RagTests


class RelevanceTests(RagTests):
    def test_gate_filters_even_exact_match_without_reordering(self):
        source = self.base / 'manual.md'
        source.write_text('# API\n\n## SUM\nSUM sums numbers.\n\n## Login\nLogin refreshes credentials.', encoding='utf-8')
        kb = self.open()
        try:
            kb.ingest(source)
            self.assertEqual(kb.search('SUM', scorer=lambda q, docs: [NS(index=i, score=.39) for i in range(len(docs))]), [])
            hits = kb.search('SUM', scorer=lambda q, docs: [NS(index=i, score=.4) for i in range(len(docs))])
            self.assertIn('/ SUM', hits[0]['path'])
        finally:
            kb.close()


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = json.loads((PROJECT / 'config/rag.json').read_text())
        config['root'] = str(self.root / 'index')
        self.memory = NS(retrieve_for_turn=AsyncMock(return_value=['memory']), format_context=lambda x: 'MEMORY')
        self.hub = RetrievalHub(self.memory, None, config)
        self.hub.rag = AsyncMock(return_value=[{'text': 'doc', 'source': 'manual'}])
        self.addCleanup(patch.stopall)
        patch('observability.setup_observability').start()
        patch('observability._TRACER', None).start()

    async def test_manual_search_has_no_hidden_call_budget(self):
        runtime = NS(config={'configurable': {'thread_id': 'code:worker'}})
        with retrieval_scope(self.hub, 'owner', 'run'):
            results = await asyncio.gather(*(search_knowledge.coroutine('query', runtime, 'both') for _ in range(4)))
        self.assertEqual([r['status'] for r in results], ['ok'] * 4)

    async def test_automatic_snapshot_independent_of_memory_and_manual_budget(self):
        with retrieval_scope(self.hub, 'owner', 'run'):
            first = await automatic_rag('code task', 'Code Agent', 'step1')
            second = await automatic_rag('repair text', 'Code Agent', 'step1')
        self.assertEqual(first, second)
        self.assertIn('[RAG使用提醒]', first)
        self.assertIn('不能代替所有项目', first)
        self.hub.rag.assert_awaited_once()
        self.memory.retrieve_for_turn.assert_not_awaited()
        self.assertEqual(await automatic_rag('private', 'Code Agent'), '')

    async def test_automatic_trace_separates_content_and_serialized_tokens(self):
        self.hub.rag = AsyncMock(return_value=[{
            'text': 'doc', 'source': 'manual', 'context_tokens': 3,
        }])
        self.hub.embedding = NS(tokenizer=NS(
            encode=lambda text, add_special_tokens=False: list(text)
        ))
        with patch('knowledge_rag.runtime.set_span_output') as record:
            with retrieval_scope(self.hub, 'owner', 'metric-run'):
                text = await automatic_rag('metric query', 'General', 'metric-step')
        payload = record.call_args.args[1]
        self.assertEqual(payload['content_tokens'], 3)
        self.assertEqual(payload['serialized_context_tokens'], len(text))
        self.assertGreater(payload['serialized_context_tokens'], payload['content_tokens'])

    async def test_failure_is_empty_no_context(self):
        self.hub.rag.side_effect = RuntimeError('scorer unavailable')
        with retrieval_scope(self.hub, 'owner', 'run'):
            self.assertEqual(await automatic_rag('task', 'Scheduler'), '')

    async def test_upload_next_message_only_idempotent_and_sender_isolated(self):
        inbox = RagUploadInbox(self.root / 'upload-mode')
        inbox.arm('chat', 'alice')
        store = NS(prepare=AsyncMock(return_value=NS(instruction='SUM sums numbers.', attachments=[])))
        self.hub.ingest = AsyncMock(return_value=[{'status': 'indexed'}])
        msg = NS(chat_id='chat', sender=NS(open_id='alice'), message_id='m1', content=NS(kind='text'))
        stranger = NS(chat_id='chat', sender=NS(open_id='bob'), message_id='m0')
        self.assertIsNone(await inbox.consume(stranger, store, None, self.hub, 'conv'))
        reply = await inbox.consume(msg, store, None, self.hub, 'conv')
        self.assertIn('已存入', reply)
        self.assertEqual(await RagUploadInbox(inbox.root).consume(msg, store, None, self.hub, 'conv'), reply)
        self.hub.ingest.assert_awaited_once()
        msg.message_id = 'm2'
        self.assertIsNone(await inbox.consume(msg, store, None, self.hub, 'conv'))

    async def test_video_consumes_mode_and_cannot_become_agent_task(self):
        inbox = RagUploadInbox(self.root / 'upload-mode')
        inbox.arm('chat', 'alice')
        store = NS(prepare=AsyncMock())
        msg = NS(chat_id='chat', sender=NS(open_id='alice'), message_id='m1', content=NS(kind='video'))
        reply = await inbox.consume(msg, store, None, self.hub, 'conv')
        self.assertIn('无法', reply)
        self.assertIn('重新点击', reply)
        store.prepare.assert_not_awaited()
        msg.message_id = 'm2'
        self.assertIsNone(await inbox.consume(msg, store, None, self.hub, 'conv'))

    async def test_registered_schema_hides_runtime_and_exposes_sources(self):
        schema = search_knowledge.tool_call_schema.model_json_schema()
        self.assertNotIn('runtime', schema['properties'])
        self.assertEqual(schema['properties']['source']['enum'], ['rag', 'memory', 'both'])
        read_schema = read_knowledge.tool_call_schema.model_json_schema()
        self.assertNotIn('runtime', read_schema['properties'])
        self.assertNotIn('parent_levels', read_schema['properties'])
        self.assertNotIn('mode', read_schema['properties'])
        self.assertEqual(read_schema['properties']['view']['enum'], ['content', 'children'])
        self.assertIn('自动RAG上下文', read_schema['properties']['reference_id']['description'])
        self.assertIn('原样复制', read_schema['properties']['node_id']['description'])
        self.assertEqual(read_schema['properties']['depth']['minimum'], 1)
        self.assertEqual(read_schema['properties']['depth']['maximum'], 8)
        self.assertIn('API', search_knowledge.tool_call_schema.model_json_schema()['properties']['source']['description'])

    async def test_real_general_graph_can_search_repeatedly(self):
        from langchain_core.messages import AIMessage, ToolMessage
        from test_code_agents import ToolRecordingModel
        from workers.general_worker import create_general_worker
        model = ToolRecordingModel(responses=[
            *[AIMessage(content='', tool_calls=[{'name':'search_knowledge', 'args':{'query':'SUM','source':'rag'}, 'id':f'search-{i}'}]) for i in range(3)],
            AIMessage(content='', tool_calls=[{'name':'report_general_result','args':{'result':{'status':'COMPLETED','summary':'retrieval verified'}},'id':'finish'}]),
        ])
        graph = create_general_worker(model, tools=[])
        with retrieval_scope(self.hub, 'owner', 'graph-run'):
            result = await graph.ainvoke({'messages':[{'role':'user','content':'查SUM'}], 'skill_mode':'off'}, config={'configurable':{'thread_id':'general'}})
        replies = [json.loads(m.content) for m in result['messages'] if isinstance(m, ToolMessage) and m.name == 'search_knowledge']
        self.assertEqual([r['status'] for r in replies], ['ok','ok','ok'])
        # No Scheduler rag_query was supplied, so Harness must not invent an
        # automatic query from the user's text. Each explicit manual call runs.
        self.assertEqual(self.hub.rag.await_count, 3)

    async def test_scheduler_wire_excludes_document_context(self):
        from planning_models import PlanningContextPack
        from scheduler_runtime import SchedulerConversation
        context = PlanningContextPack(user_request='SUM', current_time='2026-09-09', rag_context='source:manual; SUM adds numbers')
        session = SchedulerConversation(context.scheduler_session, context)
        session.initialize()
        before = session.wire()
        session.add('review', {'status':'checking'})
        self.assertEqual(session.wire()[:len(before)], before)
        self.assertFalse(any('SUM adds numbers' in m['content'] for m in before))

    async def test_code_boundary_uses_scheduler_query_without_duplicate_context(self):
        from agent import ask_worker
        from langchain_core.messages import AIMessage
        graph = NS(ainvoke=AsyncMock(return_value={'messages':[AIMessage(content='done')]}))
        with retrieval_scope(self.hub, 'owner', 'code-run'):
            assignment = (
                'long unrelated reports and budgets\n当前Step（第1次尝试）：\n'
                + json.dumps({'rag_query': 'SUM numeric total'})
            )
            await ask_worker(graph, assignment, thread_id='step-code', trace_role='code_agent', state_update={'code_task':{'requirements':[{'description':'not used as the query'}]},'execution_instructions':'fixed environment'})
        query = self.hub.rag.call_args.args[1]
        self.assertIn('SUM', query)
        self.assertNotIn('unrelated reports', query)
        actual = graph.ainvoke.call_args.args[0]['execution_instructions']
        self.assertTrue(actual.startswith('fixed environment'))
        self.assertEqual(actual, 'fixed environment')

    async def test_feishu_real_handler_upload_skips_event_queue(self):
        from test_feishu_attachments import adapter_namespace, incoming
        from feishu_attachments import FeishuAttachmentStore
        from unittest.mock import Mock
        store = FeishuAttachmentStore(self.root / 'attachments')
        runtime = NS(get_active_conversation=AsyncMock(return_value=NS(conversation_id='conv')), retrieval_hub=self.hub)
        channel = NS(download_resource=AsyncMock())
        pump = NS(notify=Mock())
        namespace = adapter_namespace(store, runtime, channel, pump)
        message = incoming('upload-1', 'SUM sums numbers.')
        namespace['RAG_UPLOADS'].arm(message.chat_id, message.sender.open_id)
        self.hub.ingest = AsyncMock(return_value=[{'status':'indexed'}])
        await namespace['handle_feishu_message'](message)
        await namespace['handle_feishu_message'](message)
        self.hub.ingest.assert_awaited_once()
        pump.notify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
