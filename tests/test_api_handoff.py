import json
from api_handoff import validate_apis
from handoff_knowledge import HandoffKnowledge,collect_handoff_knowledge

def test_flat_submission_partial_validation_and_next_step_roundtrip():
    from workers.general_completion import GeneralResult
    from handoff_knowledge import collect_api_handoffs
    from planning_models import StepReport
    submission=GeneralResult(status='COMPLETED',summary='done',handoff_apis=[{'name':'bad'},api()])
    assert len(submission.handoff_apis)==1
    trace={'general_result':submission.model_dump(mode='json'),
           'messages':[{'type':'tool','tool_call_id':k,**v} for k,v in results().items()]}
    accepted=collect_api_handoffs([trace])
    report=StepReport(step_id=1,status='COMPLETED',summary='done',stop_reason='done',handoff_apis=accepted)
    assert StepReport.model_validate_json(report.model_dump_json()).handoff_apis[0].name=='service.update'


def test_flat_handoff_reads_compacted_source_messages():
    from handoff_knowledge import collect_api_handoffs
    trace = {
        'general_result': {'handoff_apis': [api()]},
        'messages': [],
        'handoff_source_messages': [
            {'type': 'tool', 'tool_call_id': key, **value}
            for key, value in results().items()
        ],
    }
    assert [item.name for item in collect_api_handoffs([trace])] == ['service.update']


def test_flat_handoff_reads_durable_resolved_evidence():
    from handoff_knowledge import collect_api_handoffs
    trace = {
        'general_result': {'handoff_apis': [api()]},
        'worker_submission': {
            'resolved_evidence': [
                {'tool_call_id': key, 'tool_name': 'fixture',
                 'result': value['content'], 'result_chars': len(value['content'])}
                for key, value in results().items()
            ]
        },
    }
    assert [item.name for item in collect_api_handoffs([trace])] == ['service.update']

def api():
    return {'name':'service.update','purpose':'update record',
            'documentation':{'tool_call_id':'doc','pointer':'/name'},
            'parameters':[{'name':'record_id','required':True,'purpose':'target','source':{'tool_call_id':'query','pointer':'/id'}}],
            'call_example':'service.update(record_id=record_id)','next_action':'apply then verify'}

def results():
    return {'doc':{'content':json.dumps({'name':'service.update'})},'query':{'content':'{"id":7}'}}

def test_reject_one_keep_one_and_do_not_execute_example():
    bad={**api(),'call_example':"__import__('os').system('bad')"}
    accepted,errors=validate_apis([bad,api()],results())
    assert len(accepted)==1 and errors[0]['index']==0

def test_missing_source_and_error_source_rejected_independently():
    sources=results();sources['query']['status']='error'
    assert not validate_apis([api()],sources)[0]
    missing=api();missing['parameters'][0]['source']=None
    assert len(validate_apis([missing],sources)[0])==1 # declared missing; not ready to execute

def test_reviewer_cannot_remove_worker_api_and_legacy_entry_survives():
    item={'topic':'update','source':'doc','usage':'see apis','observed_result':'not executed','next_action':'continue','apis':[api()]}
    trace={'messages':[{'type':'tool','tool_call_id':k,**v} for k,v in results().items()],
           'general_result':{'handoff_knowledge':[item]},'code_review_report':{'handoff_knowledge':[]}}
    data=collect_handoff_knowledge([trace])
    assert len(data[0].apis)==1
    assert data[0].apis[0].parameters[0].source.tool_call_id=='query'

def test_invalid_shape_does_not_drop_other_api():
    item=HandoffKnowledge(topic='update',source='doc',usage='see apis',observed_result='not executed',next_action='continue',apis=[{'name':'bad'},api()])
    assert len(item.apis)==1

def test_batch_document_stream_validates_each_api_independently():
    documents='\n'.join([
        json.dumps({'app_name':'service','api_name':'create','parameters':[]}),
        json.dumps({'app_name':'service','api_name':'update','parameters':[]}),
    ])
    sources=results()
    sources['batch']={'content':documents}
    create={**api(),'name':'service.create','documentation':{'tool_call_id':'batch','pointer':''},
            'call_example':'service.create(record_id=record_id)'}
    update={**api(),'documentation':{'tool_call_id':'batch','pointer':''}}
    missing={**api(),'name':'service.remove','documentation':{'tool_call_id':'batch','pointer':''},
             'call_example':'service.remove(record_id=record_id)'}
    accepted,errors=validate_apis([create,missing,update],sources)
    assert [item.name for item in accepted]==['service.create','service.update']
    assert errors[0]['index']==1
    assert errors[0]['name']=='service.remove'
    assert 'exactly one matching API document' in errors[0]['detail']

def test_wrapped_batch_document_stream_is_supported():
    documents='\n'.join([
        json.dumps({'app_name':'service','api_name':'create'}),
        json.dumps({'app_name':'service','api_name':'update'}),
    ])
    sources=results()
    sources['batch']={'content':json.dumps({'content':documents})}
    item={**api(),'documentation':{'tool_call_id':'batch','pointer':''}}
    assert [entry.name for entry in validate_apis([item],sources)[0]]==['service.update']

def test_call_example_accepts_runtime_apis_namespace():
    item={**api(),'call_example':'apis.service.update(record_id=record_id)'}
    assert [entry.name for entry in validate_apis([item],results())[0]]==['service.update']

def test_call_example_accepts_alias_placeholders_and_numeric_values():
    item={**api(),'documentation':{'tool_call_id':'doc','pointer':''},
          'call_example':'apis.service.update(record_id=rid, limit=20)'}
    sources=results()
    sources['doc']={'content':json.dumps({'name':'service.update','parameters':[{'name':'record_id'},{'name':'limit'}]})}
    assert [entry.name for entry in validate_apis([item],sources)[0]]==['service.update']

def test_call_example_rejects_undocumented_keyword_and_string_value():
    unknown={**api(),'call_example':'service.update(record_id=rid, surprise=1)'}
    secret={**api(),'call_example':"service.update(record_id='copied-secret')"}
    assert not validate_apis([unknown],results())[0]
    assert not validate_apis([secret],results())[0]


def test_parameter_source_accepts_successful_plain_text_when_pointer_is_empty():
    item = api()
    item["parameters"][0]["source"] = {"tool_call_id": "query", "pointer": ""}
    sources = results()
    sources["query"] = {"content": "target-id: 7\nverified"}
    accepted, errors = validate_apis([item], sources)
    assert [entry.name for entry in accepted] == ["service.update"]
    assert errors == []


def test_parameter_pointer_still_requires_structured_single_root():
    item = api()
    sources = results()
    sources["query"] = {"content": "target-id: 7\nverified"}
    accepted, errors = validate_apis([item], sources)
    assert accepted == []
    assert "JSON" in errors[0]["detail"]


def test_rejected_api_handoff_is_preserved_with_reason():
    from handoff_knowledge import collect_api_handoff_receipts
    trace = {
        "general_result": {"handoff_apis": [api()]},
        "messages": [
            {"type": "tool", "tool_call_id": "doc", "content": '{"name":"other.api"}'},
            {"type": "tool", "tool_call_id": "query", "content": '{"id":7}'},
        ],
    }
    receipts = collect_api_handoff_receipts([trace])
    assert len(receipts) == 1
    assert receipts[0].api.name == "service.update"
    assert receipts[0].validation_status == "REJECTED"
    assert "differs" in receipts[0].validation_error


def test_code_reviewer_handoff_is_not_silently_ignored():
    from handoff_knowledge import collect_api_handoff_receipts
    trace = {
        "code_review_report": {"handoff_apis": [api()]},
        "messages": [
            {"type": "tool", "tool_call_id": key, **value}
            for key, value in results().items()
        ],
    }
    receipts = collect_api_handoff_receipts([trace])
    assert receipts[0].validation_status == "VALIDATED"


def test_non_document_source_reports_clear_validation_error():
    sources = results()
    sources["doc"] = {"content": "created record 17"}
    accepted, errors = validate_apis([api()], sources)
    assert accepted == []
    assert errors[0]["detail"] == "documentation source is not API documentation JSON"
