import json
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from workers.evidence_refs import registry, message_view, canonical_arguments, schema_view, EvidenceReferenceMiddleware
from workers.general_completion import report_general_result
from workers.submission import submit_for_review, resolve_tool_evidence
from workers.code_submission import submit_code_for_review, respond_to_code_review, submit_continued_code_for_review, submit_code_review

RAW='call_1234567890abcdefghijklmnopqrstuvwxyz'
def history():
    return [AIMessage(content='',tool_calls=[{'id':RAW,'name':'probe','args':{}}]),
            ToolMessage(content='actual result',name='probe',tool_call_id=RAW)]

def test_stable_archive_mapping_and_separate_agent_scope():
    old=history(); refs=registry({'messages':old})
    assert refs=={RAW:'E1'}
    new=AIMessage(content='',tool_calls=[{'id':'next-call','name':'probe','args':{}}])
    resumed={'worker_evidence_refs':refs,'worker_archived_messages':old,'messages':[new]}
    assert registry(resumed)=={RAW:'E1','next-call':'E2'}
    assert refs=={RAW:'E1'}
    assert registry({'messages':[]})=={}
    # Receiver imports canonical evidence, never another agent's local E1.
    inherited={'code_worker_submission':{'resolved_evidence':[{'tool_call_id':RAW,'tool_name':'probe','result':'actual'}]}}
    assert registry(inherited)=={RAW:'E1'}

def test_views_hide_ids_without_mutating_canonical_messages():
    originals=history(); refs=registry({'messages':originals})
    viewed=[message_view(m,refs) for m in originals]
    assert viewed[0].tool_calls[0]['id']==viewed[1].tool_call_id=='E1'
    assert RAW not in json.dumps([m.model_dump() for m in viewed])
    assert originals[1].tool_call_id==RAW
    restored=canonical_arguments({'claims':[{'evidence_refs':['E1']}], 'files':[{'evidence_ref':'E1'}]},refs)
    assert restored['claims'][0]['evidence_tool_call_ids']==[RAW]
    assert restored['files'][0]['evidence_tool_call_id']==RAW
    assert resolve_tool_evidence(originals,restored['claims'][0]['evidence_tool_call_ids'])[0].result=='actual result'

@pytest.mark.parametrize('tool',[report_general_result,submit_for_review,submit_code_for_review,respond_to_code_review,submit_continued_code_for_review,submit_code_review])
def test_all_submission_schemas_hide_long_id_contract(tool):
    before=convert_to_openai_tool(tool)
    viewed=convert_to_openai_tool(schema_view(tool))
    assert 'evidence_tool_call_ids' not in json.dumps(viewed)
    assert before==convert_to_openai_tool(tool)
    if 'evidence_tool_call_ids' in json.dumps(before): assert 'evidence_refs' in json.dumps(viewed)

def test_invalid_or_foreign_reference_rejected_without_tool_execution():
    with pytest.raises(ValueError,match='Unknown evidence'):
        canonical_arguments({'evidence_refs':['E2']},{RAW:'E1'})
    with pytest.raises(ValueError,match='Duplicate'):
        canonical_arguments({'evidence_refs':['E1'],'evidence_tool_call_ids':[RAW]},{RAW:'E1'})

def test_reviewer_existing_field_uses_same_mapping_and_rejects_control_evidence():
    assert canonical_arguments({'report':{'evidence_refs':['E1']}},{RAW:'E1'},reviewer=True,eligible={RAW}) == {'report':{'evidence_refs':[RAW]}}
    with pytest.raises(ValueError,match='completed business'):
        canonical_arguments({'report':{'evidence_refs':['E1']}},{RAW:'E1'},reviewer=True,eligible=set())

def test_progress_references_are_not_redefined_as_tool_ids():
    from workers.progress import publish_worker_progress
    original=convert_to_openai_tool(publish_worker_progress)
    assert convert_to_openai_tool(schema_view(publish_worker_progress))==original
    assert canonical_arguments({'evidence_refs':['E1','https://example.com']},{RAW:'E1'},progress=True,eligible={RAW}) == {'evidence_refs':[RAW,'https://example.com']}

def test_compression_model_sees_short_ids_and_mapping_is_checkpointable():
    from workers.compaction import WorkerCompactionMiddleware
    refs=registry({'messages':history()})
    request=WorkerCompactionMiddleware(None)._request(history(),[0,1],refs)
    assert RAW not in json.dumps(request)
    assert 'E1' in json.dumps(request)
    assert json.loads(json.dumps(refs))==refs

def test_web_real_graph_short_refs():
    from test_general_simple_flow import GeneralModel
    from test_worker_submission import web_search
    from langchain_core.outputs import ChatGeneration,ChatResult
    from workers.web_worker import create_web_worker
    class WebModel(GeneralModel):
        def _generate(self,messages,stop=None,run_manager=None,**kwargs):
            if not any(isinstance(m,ToolMessage) for m in messages):
                msg=AIMessage(content='',tool_calls=[{'name':'web_search','id':RAW,'args':{'query':'check'}}])
            else:
                msg=AIMessage(content='',tool_calls=[{'name':'submit_for_review','id':'submission','args':{'submission':{'summary':'checked','final_conclusion':'observed','criterion_claims':[{'criterion':'check','conclusion':'observed','evidence_refs':['E1']}]}}}])
            return ChatResult(generations=[ChatGeneration(message=msg)])
    result=create_web_worker(WebModel(),tools=[web_search]).invoke({'messages':[HumanMessage(content='check')],'worker_id':'web'})
    record=result['worker_submission']
    assert record['resolved_evidence'][0]['tool_call_id']==RAW
    assert record['submission']['criterion_claims'][0]['evidence_tool_call_ids']==[RAW]

def test_code_real_graph_short_refs():
    from test_general_simple_flow import GeneralModel
    from test_code_agents import candidate,contract,code_probe
    from workers.code_worker import create_code_worker
    from langchain_core.outputs import ChatGeneration,ChatResult
    class CodeModel(GeneralModel):
        def _generate(self,messages,stop=None,run_manager=None,**kwargs):
            if not any(isinstance(m,ToolMessage) for m in messages):
                msg=AIMessage(content='',tool_calls=[{'name':'code_probe','id':RAW,'args':{'value':'check'}}])
            else:
                msg=AIMessage(content='',tool_calls=[{'name':'submit_code_for_review','id':'submission','args':{'submission':{'candidate':candidate().model_dump(mode='json'),'summary':'checked','requirement_status':{'feature_works':'MET'},'evidence_refs':['E1']}}}])
            return ChatResult(generations=[ChatGeneration(message=msg)])
    result=create_code_worker(CodeModel(),tools=[code_probe]).invoke({'messages':[HumanMessage(content='check')],'worker_id':'code','code_task':contract().model_dump(mode='json'),'code_candidate':candidate().model_dump(mode='json')})
    assert result['code_worker_submission']['resolved_evidence'][0]['tool_call_id']==RAW
    assert result['code_worker_submission']['submission']['evidence_tool_call_ids']==[RAW]

def test_mapping_in_trace_attributes_not_prompt_options():
    from trace_callbacks import RuntimeTraceCallback
    from workers.evidence_refs import ACTIVE_EVIDENCE_REFS
    callback=RuntimeTraceCallback()
    captured=[]
    callback._start=lambda *args: captured.append(args)
    token=ACTIVE_EVIDENCE_REFS.set({RAW:'E1'})
    try: callback.on_chat_model_start({},[[HumanMessage(content='hello')]],run_id='offline',invocation_params={})
    finally: ACTIVE_EVIDENCE_REFS.reset(token)
    assert json.loads(captured[0][-1]['evidence.references_json'])=={'E1':RAW}
    assert RAW not in json.dumps(captured[0][-2])

def test_code_reviewer_real_graph_short_refs():
    from test_general_simple_flow import GeneralModel
    from test_code_agents import candidate,contract,code_probe
    from workers.code_reviewer import create_code_reviewer
    from workers.code_review_models import create_code_review_loop
    from langchain_core.outputs import ChatGeneration,ChatResult
    class ReviewerModel(GeneralModel):
        def _generate(self,messages,stop=None,run_manager=None,**kwargs):
            if not any(isinstance(m,ToolMessage) for m in messages):
                msg=AIMessage(content='',tool_calls=[{'name':'code_probe','id':RAW,'args':{'value':'check'}}])
            else:
                msg=AIMessage(content='',tool_calls=[{'name':'submit_code_review','id':'review','args':{'report':{'candidate':candidate().model_dump(mode='json'),'verdict':'FAILED','summary':'Needs work','verification_summary':'Checked','recommended_action':'STOP','evidence_refs':['E1']}}}])
            return ChatResult(generations=[ChatGeneration(message=msg)])
    result=create_code_reviewer(ReviewerModel(),tools=[code_probe]).invoke({'messages':[HumanMessage(content='review')],'worker_id':'reviewer','code_task':contract().model_dump(mode='json'),'code_candidate':candidate().model_dump(mode='json'),'code_review_loop':create_code_review_loop(candidate=candidate(),worker_checkpoint_id="worker",reviewer_checkpoint_id="reviewer").model_dump(mode='json')})
    assert result['code_review_report']['evidence_refs']==[RAW]

def test_general_real_graph_accepts_short_refs_and_persists_canonical_evidence():
    import asyncio
    from test_general_simple_flow import GeneralModel,business_probe
    from workers.general_worker import create_general_worker
    class ShortModel(GeneralModel):
        def _generate(self,messages,stop=None,run_manager=None,**kwargs):
            result=super()._generate(messages,stop,run_manager,**kwargs)
            for generation in result.generations:
                for call in generation.message.tool_calls:
                    if call['name']=='report_general_result':
                        payload=call['args']['result']
                        payload.pop('evidence_tool_call_ids',None)
                        payload['evidence_refs']=['E1']
            return result
    model=ShortModel()
    result=asyncio.run(create_general_worker(model,tools=[business_probe]).ainvoke({'messages':[HumanMessage(content='Check')],'worker_id':'general','event_id':'evt','step_id':'1'}))
    assert result['general_result']['evidence_tool_call_ids']==['probe']
    assert result['worker_evidence_refs']['probe']=='E1'
