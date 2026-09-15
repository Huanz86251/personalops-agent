"""Capture five real task contexts without a provider or evaluator access.

This is a preflight, deliberately stopping at the first model request. It is
not an AppWorld solve, a grade, or a Code Agent acceptance test.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CapturedRequest(BaseException):
    """Exit outside runtime repair handling: no failed model response exists."""


async def main():
    out = ROOT / '.agent/appworld-contexts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(PHOENIX_TRACING_ENABLED='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    logging.basicConfig(filename=out / 'runtime.private.log', encoding='utf-8', level=logging.INFO)
    import path
    path.AGENT_DATA_ROOT = out / 'state'
    path.WORKSPACE_ROOT = out / 'workspace'
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.runnables import RunnableLambda
    from evals.appworld.protocol import DockerWorld
    from evals.appworld.adapter import run_personalops

    # Docker is a subprocess transport. No Python provider/network requests are allowed.
    original_connect = socket.socket.connect
    def blocked_connect(*args, **kwargs):
        raise AssertionError('Network disabled for context preflight')
    socket.socket.connect = blocked_connect
    summary = []
    try:
        with DockerWorld() as world:
            ids = world.request('list_tasks', split='train')
        selected = []
        families = set()
        for task_id in ids:
            family = task_id.split('_')[0]
            if family not in families:
                selected.append(task_id)
                families.add(family)
            if len(selected) == 5:
                break
        assert len(selected) == 5
        for index, task_id in enumerate(selected, 1):
            folder = out / f'{index:02d}'
            folder.mkdir()
            def save(name, data):
                (folder / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            class CaptureModel(BaseChatModel):
                @property
                def _llm_type(self):
                    return 'offline-context-capture'
                def bind_tools(self, tools, **kwargs):
                    return self
                def with_structured_output(self, schema, **kwargs):
                    async def capture(messages, config=None):
                        save('request.private.json', {'schema': schema.model_json_schema(),
                             'messages': [m if isinstance(m, dict) else m.model_dump(mode='json') for m in messages]})
                        raise CapturedRequest()
                    return RunnableLambda(capture)
                def _generate(self, *args, **kwargs):
                    raise AssertionError('Unexpected unstructured model call')
            with DockerWorld() as world:
                task = world.request('initialize', split='train', task_id=task_id,
                                     trial_id=f'context_{out.name}_{index}', max_interactions=10)
                save('task.private.json', task)
                save('isolation.json', world.isolation_report())
                save('apps.private.json', world.execute('print(apis.api_docs.show_app_descriptions())'))
                model = CaptureModel()
                try:
                    await run_personalops(world, task, simple_model=model, hard_model=model,
                                          trial_id=f'context_{out.name}_{index}', wall_timeout=60)
                except CapturedRequest:
                    pass
                assert (folder / 'request.private.json').exists()
                assert world.request('task_completed') is False
            request = json.loads((folder / 'request.private.json').read_text(encoding='utf-8'))
            summary.append({'case': index, 'task_id': task_id, 'status': 'paused_before_first_model_response',
                            'message_count': len(request['messages']),
                            'message_chars': sum(len(json.dumps(m, ensure_ascii=False)) for m in request['messages']),
                            'paid_requests': 0, 'graded': False})
            print(json.dumps(summary[-1]), flush=True)
    finally:
        socket.socket.connect = original_connect
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(str(out), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
