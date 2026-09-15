from langchain_core.messages import HumanMessage
from model_clients import CompatibleChatModel
import pytest

TOOL = {'type':'function','function':{'name':'submit_report','description':'Submit result',
        'parameters':{'type':'object','properties':{}}}}

@pytest.mark.parametrize('choice',['required',{'type':'function','function':{'name':'submit_report'}}])
def test_forced_handoff_retains_tools_messages_and_disables_only_request_thinking(choice):
    extra={'enable_thinking':True,'reasoning_effort':'low','preserve_thinking':False}
    model=CompatibleChatModel(model='qwen3.8-flash',api_key='offline',extra_body=extra)
    messages=[HumanMessage(content='Submit actual results')]
    normal=model._get_request_payload(messages,tools=[TOOL],tool_choice='auto')
    forced=model._get_request_payload(messages,tools=[TOOL],tool_choice=choice)
    assert forced['tool_choice']==choice
    assert forced['tools']==normal['tools']
    assert forced['messages']==normal['messages']
    assert forced['extra_body']=={'enable_thinking':False,'preserve_thinking':False}
    assert model._get_request_payload(messages,tool_choice='auto')['extra_body']==extra
    assert model.extra_body==extra
    recorded=model._get_invocation_params(tools=[TOOL],tool_choice=choice)
    assert recorded['extra_body']==forced['extra_body']

@pytest.mark.parametrize('model_name',['qwen3.7-flash','unrelated-model'])
def test_other_models_unchanged(model_name):
    extra={'enable_thinking':True,'thinking_budget':1024}
    model=CompatibleChatModel(model=model_name,api_key='offline',extra_body=extra)
    assert model._get_request_payload('x',tool_choice='required')['extra_body']==extra

def test_bound_named_tool_and_top_level_effort():
    model=CompatibleChatModel(model='qwen3.8-flash',api_key='offline',reasoning_effort='low',
                              extra_body={'enable_thinking':True,'thinking_budget':4096})
    bound=model.bind_tools([TOOL],tool_choice='submit_report')
    payload=model._get_request_payload('x',**bound.kwargs)
    assert payload['tool_choice']['function']['name']=='submit_report'
    assert payload['extra_body']=={'enable_thinking':False}
    assert 'reasoning_effort' not in payload
