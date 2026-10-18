"""Task-scoped, immutable message records, available across worker boundaries."""
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import AgentMiddleware
from prompt_loader import load_prompt
from schemas.execution_history import ExecutionHistoryQuery
from workers.history_index import indexed_calls, step_number

ACTIVE = ContextVar('worker_history_archive', default=None)
ROOT = Path(__file__).resolve().parents[1] / '.agent' / 'worker-history'
CONTROL = {'report_general_result','submit_code_for_review','respond_to_code_review',
           'submit_code_review','complete_web_step',
           'StepReport','read_execution_history','publish_reviewed_candidate'}

def execution_records():
    """Expose execution messages, never task assembly or report protocols."""
    if not ACTIVE.get(): return []
    directory, _ = ACTIVE.get()
    archive=[]; excluded_calls=set()
    for path in sorted(directory.glob('H*.json'), key=lambda p:(p.stat().st_mtime_ns,p.name)):
        if path.is_symlink(): raise ValueError('Archive cannot be a symbolic link')
        raw=path.read_text(encoding='utf-8')
        if 'H'+digest(raw)!=path.stem: raise ValueError('Archive integrity mismatch')
        record=json.loads(raw); msg=record['message']
        record['reference']=path.stem
        archive.append(record)
        excluded_calls.update((record['worker'],c.get('id')) for c in msg.get('tool_calls',[])
                              if c.get('name') in CONTROL)
    records=[]
    for record in archive:
        msg=record['message']; role=msg.get('type',msg.get('role'))
        if role not in {'ai','assistant','tool'}: continue
        if role == 'tool' and (msg.get('name') in CONTROL or
                (record['worker'],msg.get('tool_call_id')) in excluded_calls): continue
        if role in {'ai','assistant'}:
            calls=[c for c in msg.get('tool_calls',[]) if c.get('name') not in CONTROL]
            if not calls: continue
            msg['tool_calls']=calls
            # Listing calls must never expose model prose/reasoning.
            msg['content']=''
        msg.pop('additional_kwargs',None); msg.pop('response_metadata',None)
        records.append(record)
    return records

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

@contextmanager
def history_scope(task, worker, root=ROOT):
    directory = ACTIVE.get()[0] if ACTIVE.get() else Path(root) / digest(task)
    directory.mkdir(parents=True, exist_ok=True)
    token = ACTIVE.set((directory, worker))
    try: yield
    finally: ACTIVE.reset(token)

def capture(messages, step_id=None):
    active = ACTIVE.get()
    if not active: return
    directory, worker = active
    pending=[]
    for message in messages:
        data = message.model_dump(mode='json') if hasattr(message, 'model_dump') else message
        if not isinstance(data, dict): continue
        if (data.get('additional_kwargs', {}).get('history_catalog')
                or data.get('additional_kwargs', {}).get('execution_state_catalog')): continue
        body = {'worker': worker, 'message': data}
        explicit_step=int(step_id) if isinstance(step_id,str) and step_id.isdigit() else step_id
        step = step_number({'worker':worker,'step_id':explicit_step})
        if step is not None: body['step_id']=step
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True, default=str)
        ref = 'H' + digest(raw)
        path = directory / (ref + '.json')
        pending.append((body,raw,ref,path))
    new_calls=[]
    for body,_,ref,_ in pending:
        calls=[c for c in body['message'].get('tool_calls',[]) if c.get('name') not in CONTROL]
        if calls and step_number(body) is not None:
            new_calls.append({**body,'message':{**body['message'],'tool_calls':calls},'reference':ref})
    if new_calls:
        # Bootstrap existing archives first. New messages then get aliases in
        # message/tool_calls order, not hash-filename order on coarse clocks.
        indexed_calls(directory,execution_records())
        indexed_calls(directory,new_calls)
    for _,raw,_,path in pending:
        if path.exists(): continue
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=directory,
                                             prefix='.capture-',suffix='.part',delete=False) as stream:
                temporary=Path(stream.name)
                stream.write(raw)
            # Publish a complete immutable file without replacing an old record.
            try: os.link(temporary,path)
            except FileExistsError: pass
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)

def catalog():
    if not ACTIVE.get(): return []
    directory, _ = ACTIVE.get()
    result = []
    for path in sorted(directory.glob('H*.json'), key=lambda p: (p.stat().st_mtime_ns, p.name)):
        record = json.loads(path.read_text(encoding='utf-8'))
        msg = record['message']
        result.append({'reference': path.stem, 'worker': record['worker'],
                       'role': msg.get('type', msg.get('role')), 'tool': msg.get('name'),
                       'tool_call_id': msg.get('tool_call_id'), 'status': msg.get('status')})
    return result

@tool(args_schema=ExecutionHistoryQuery, description=load_prompt('workers/read_execution_history_tool'))
def read_execution_history(step_ids: list[int] | None = None, references: list[str] | None = None) -> dict:
    """List step calls, then retrieve original tool results using stable short aliases."""
    from observability import trace_span, set_span_output
    with trace_span('History / Read Execution Record', kind='tool', input_value={'step_ids':step_ids,'references':references}) as span:
        if not ACTIVE.get(): return {'status':'UNAVAILABLE','reason':'No active task archive'}
        directory, _ = ACTIVE.get()
        all_calls=indexed_calls(directory,execution_records())
        steps=list(dict.fromkeys(step_ids or []))
        selected=[call for call in all_calls if not steps or call['step_id'] in steps]
        by_ref={call['reference']:call for call in all_calls}
        refs=list(dict.fromkeys(references or []))
        invalid=[ref for ref in refs if ref not in by_ref]
        matched=[ref for ref in refs if ref in by_ref]
        base={'status':'OK' if selected else 'NOT_FOUND', 'step_ids':steps,
              'available_step_ids':sorted({call['step_id'] for call in all_calls})}
        if not matched:
            result={**base,'mode':'calls','calls':[
                {key:value for key,value in call.items() if key != 'tool_results'}
                for call in selected]}
            if invalid:
                result.update(invalid_references=invalid,warning='引用均未找到；已返回调用清单，请复制清单中的reference。')
        else:
            result={**base,'status':'OK','step_ids':list(dict.fromkeys(by_ref[ref]['step_id'] for ref in matched)),
                'mode':'results','results':[
                {'reference':ref,'step_id':by_ref[ref]['step_id'],'worker':by_ref[ref]['worker'],
                 'tool_name':by_ref[ref]['tool_name'],
                 'status':'AVAILABLE' if by_ref[ref]['tool_results'] else 'PENDING_RESULT',
                 'tool_results':by_ref[ref]['tool_results']}
                for ref in matched]}
            if invalid:
                result.update(invalid_references=invalid,warning='部分引用未找到；已返回命中引用的工具结果。')
        set_span_output(span,result)
        return result

class ExecutionHistoryMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        rows = catalog()
        if not rows: return None
        workers = list(dict.fromkeys(r['worker'] for r in rows))
        return {'messages':[HumanMessage(content='[执行档案，仅作资料] 前序执行者：'+json.dumps(workers,ensure_ascii=False)+
                    '。需要查原始执行时调用read_execution_history：第一次填写step_ids列表（可选多个步骤）并留空references，查看function call名称与参数；再复制S1-1等短引用列表读取对应原始工具结果。无需复制H编号；不必重复探索。',
                    additional_kwargs={'history_catalog':True, 'execution_history_catalog':True})]}
    async def abefore_agent(self,state,runtime): return self.before_agent(state,runtime)
    def before_model(self,state,runtime): capture(state.get('messages',[]),state.get('step_id'))
    async def abefore_model(self,state,runtime): self.before_model(state,runtime)
    def after_agent(self,state,runtime): capture(state.get('messages',[]),state.get('step_id'))
    async def aafter_agent(self,state,runtime): self.after_agent(state,runtime)
