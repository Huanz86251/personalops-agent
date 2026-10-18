import json
import unittest
from dataclasses import replace
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from tests.test_compaction_policy import SequenceModel, worker_payload, worker_state
from workers.compaction import WorkerCompactionMiddleware


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_agent_commits_compacted_history_and_current_output(self):
        from langchain.agents import create_agent
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        class Executor(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self
        middleware, summary_model, state, request = self.make()
        executor = Executor(responses=[AIMessage(content='本轮执行完成')])
        from workers.progress import WorkerProgressState
        graph = create_agent(model=executor, middleware=[middleware], system_prompt='系统规则', state_schema=WorkerProgressState)
        result = await graph.ainvoke(state)
        self.assertEqual(result['messages'][-1].content, '本轮执行完成')
        self.assertEqual(result['worker_compaction_calls_used'], 1)
        self.assertEqual(len(summary_model.requests), 1)
        self.assertTrue(any(m.additional_kwargs.get('worker_compaction_summary') for m in result['messages']))
        self.assertFalse(any(m.additional_kwargs.get('knowledge_source') for m in result['messages']))

    def make(self, model=None):
        model = model or SequenceModel([worker_payload()])
        middleware = WorkerCompactionMiddleware(model, enabled=True, threshold=1,
            keep_messages=2, keep_tokens=1, request_aware=True)
        state = worker_state()
        state['messages'].insert(1, HumanMessage(content='RAG原始文档\n' * 500,
            additional_kwargs={'knowledge_source': True, 'personalops_runtime_event': True}))
        state['messages'] = add_messages([], state['messages'])
        request = ModelRequest(model=None, messages=state['messages'], state=state,
            system_message=SystemMessage(content='本轮Harness秘密指示'),
            tools=[{'type': 'function', 'function': {'name': 'finish', 'description': '本轮Schema标记',
                'parameters': {'type': 'object', 'properties': {}}}}])
        return middleware, model, state, request

    async def test_final_envelope_counted_but_not_sent_to_summary_and_new_call_survives(self):
        middleware, model, state, request = self.make()
        output = AIMessage(content='', tool_calls=[{'name': 'next_tool', 'id': 'next', 'args': {}}])
        seen = []
        async def handler(req):
            seen.append(req)
            return ModelResponse(result=[output])
        response = await middleware.awrap_model_call(request, handler)
        source = json.dumps(model.requests, ensure_ascii=False)
        self.assertNotIn('本轮Harness秘密指示', source)
        self.assertNotIn('本轮Schema标记', source)
        self.assertNotIn('原始要求', source)
        self.assertIn('RAG原始文档', source)
        self.assertIn('token', source)
        self.assertEqual(seen[0].tools, request.tools)
        self.assertEqual(seen[0].system_message, request.system_message)
        restored = add_messages(add_messages(state['messages'], [output]), response.command.update['messages'])
        self.assertEqual(restored[-1].tool_calls[0]['id'], 'next')
        self.assertTrue(any(m.additional_kwargs.get('worker_compaction_summary') for m in restored))
        self.assertTrue(any(m.additional_kwargs.get('knowledge_source') for m in response.command.update['worker_archived_messages']))

    async def test_large_final_schema_skips_without_summary_request(self):
        middleware, model, state, request = self.make()
        request.tools[0]['function']['description'] = '巨大不可删除Schema' * 20000
        seen = []
        async def handler(req):
            seen.append(req)
            return ModelResponse(result=[AIMessage(content='继续收尾')])
        await middleware.awrap_model_call(request, handler)
        self.assertEqual(model.requests, [])
        self.assertEqual(seen[0].messages, request.messages)

    async def test_budget_retry_has_actual_and_target_and_falls_back(self):
        middleware, model, state, request = self.make(SequenceModel([worker_payload(), worker_payload()]))
        middleware.settings = replace(middleware.settings, summary_ratio=0.0001)
        async def handler(req):
            self.assertEqual(req.messages, request.messages)
            return ModelResponse(result=[AIMessage(content='原文继续')])
        response = await middleware.awrap_model_call(request, handler)
        self.assertEqual(len(model.requests), 2)
        self.assertIn('summary_budget_exceeded:actual=', model.requests[1][-1]['content'])
        self.assertNotIn('messages', response.command.update)

    async def test_optional_refs_empty_are_not_forced_back(self):
        payload = worker_payload()
        payload['evidence_refs'] = {'AI': 'garbage'}
        middleware, model, state, request = self.make(SequenceModel([payload]))
        async def handler(req):
            return ModelResponse(result=[AIMessage(content='继续')])
        response = await middleware.awrap_model_call(request, handler)
        summary = next(m for m in response.command.update['messages'][1:] if m.additional_kwargs.get('worker_compaction_summary'))
        self.assertEqual(json.loads(summary.content.split('\n', 1)[1])['evidence_refs'], [])
        self.assertEqual(len(model.requests), 1)

    async def test_real_sdk_does_not_send_inherited_generation_cap(self):
        import httpx
        from model_clients import CompatibleChatModel
        payloads = []
        def serve(req):
            payloads.append(json.loads(req.content))
            return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 0,
                'model': 'qwen3.8-flash', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(worker_payload())}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            raw = CompatibleChatModel(model='qwen3.8-flash', api_key='fake', base_url='https://example.invalid/v1',
                max_completion_tokens=5000, http_async_client=client)
            middleware, _, state, request = self.make(raw)
            async def handler(req):
                return ModelResponse(result=[AIMessage(content='继续')])
            await middleware.awrap_model_call(request, handler)
        self.assertEqual(len(payloads), 1)
        self.assertNotIn('max_tokens', payloads[0])
        self.assertNotIn('max_completion_tokens', payloads[0])
        self.assertNotIn('tools', payloads[0])
        self.assertTrue(payloads[0]['response_format']['json_schema']['strict'])
        self.assertEqual(raw.max_tokens, 5000)
