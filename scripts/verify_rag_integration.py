"""Local models + real retrieval/tool graph; no paid model or real Feishu send."""
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]


async def main():
    from sentence_transformers import CrossEncoder
    from retrieval_models import RetrievalModelManager
    from knowledge_rag.runtime import RetrievalHub, retrieval_scope, automatic_rag, search_knowledge
    from knowledge_rag.service import PROJECT
    from observability import trace_span, set_span_output
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from openinference.instrumentation import OITracer, TraceConfig
    from unittest.mock import AsyncMock
    import observability

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    directory = ROOT / '.agent/rag-integration-checks' / stamp
    directory.mkdir(parents=True)
    project = 'RAG 接线验证 · ' + stamp
    provider = TracerProvider(resource=Resource.create({'openinference.project.name':project}))
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint='http://127.0.0.1:6007/v1/traces', timeout=5)))
    observability._TRACER = OITracer(provider.get_tracer('rag-local-check'), config=TraceConfig())

    manager = RetrievalModelManager('unused', 'maidalun1020/bce-reranker-base_v1', ROOT/'.models', device='cuda')
    manager._reranker_model = CrossEncoder(manager.reranker_model_name, cache_folder=str(ROOT/'.models'), device='cuda', local_files_only=True, max_length=1024)
    config = json.loads((PROJECT/'config/rag.json').read_text())
    config['root'] = str(directory/'index')
    memory = SimpleNamespace(retrieve_for_turn=AsyncMock(return_value=[]), format_context=lambda values:'')
    hub = RetrievalHub(memory, manager, config)
    doc = directory/'manual.md'
    doc.write_text('# API manual\n\n## SUM\nSUM adds numeric values to compute a total.\n\n## Spotify playlists\nUse documented Spotify APIs to read all pages of songs from playlists before changing ratings. Only change songs the user liked.\n\n## Login\nRefresh expired access tokens before accessing records.', encoding='utf-8')
    with trace_span('Local RAG Acceptance', input_value={'paid_model_requests':0,'fixture':'authored documentation; not AppWorld score'}) as root_span:
        indexed = await hub.ingest('test', [doc])
        with retrieval_scope(hub, 'test', stamp):
            with trace_span('Scheduler', kind='agent'):
                scheduler = await automatic_rag('怎样获取Spotify歌单里的歌曲并修改喜欢歌曲的评分？', 'Scheduler')
                with trace_span('Code Agent / Step 1', kind='agent'):
                    code = await automatic_rag('如何使用SUM把数值求和？', 'Code Agent', 'step1')
                    runtime = SimpleNamespace(config={'configurable':{'thread_id':'code-worker'}})
                    calls = [await search_knowledge.coroutine('SUM numeric total',runtime,'rag') for _ in range(3)]
            unrelated = await hub.rag('test', '火星岩石的同位素年龄是多少？')
        result = {'index':indexed, 'scheduler_injected':bool(scheduler), 'code_injected':bool(code), 'manual_statuses':[c['status'] for c in calls], 'unrelated_hits':len(unrelated),'paid_requests':0}
        result['passed'] = bool(scheduler and code and not unrelated and result['manual_statuses']==['ok','ok','budget_exhausted'])
        set_span_output(root_span,result)
    spans = exporter.get_finished_spans()
    result['project'] = project
    result['trace_id'] = f'{spans[-1].context.trace_id:032x}'
    result['spans'] = [{'name':s.name,'id':f'{s.context.span_id:016x}','parent':f'{s.parent.span_id:016x}' if s.parent else None,'attributes':dict(s.attributes)} for s in spans]
    ids={s['id'] for s in result['spans']}
    result['missing_parents']=[s['id'] for s in result['spans'] if s['parent'] and s['parent'] not in ids]
    (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    provider.shutdown()
    manager.close()
    print(json.dumps({k:v for k,v in result.items() if k!='spans'},ensure_ascii=False))
    print(directory)
    if not result['passed'] or result['missing_parents']:
        raise RuntimeError('Inspect the local RAG evidence')


if __name__=='__main__':
    asyncio.run(main())
