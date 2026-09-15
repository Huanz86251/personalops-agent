import copy
from langchain_core.messages import AIMessageChunk
from openai.types.chat import ChatCompletion
from model_clients import CompatibleChatModel
from trace_chat import chat_attributes, message_record


def response():
    return {'id':'test-response','object':'chat.completion','created':1,'model':'test',
        'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
          'reasoning_content':'Synthetic reasoning fragment.',
          'tool_calls':[{'id':'call1','type':'function','function':{'name':'probe','arguments':'{}'}}]}}],
        'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15,
                 'completion_tokens_details':{'reasoning_tokens':3}}}


def test_provider_body_dict_and_sdk_survive_conversion_without_prompt_growth():
    model=CompatibleChatModel(model='test',api_key='offline-placeholder')
    for raw in (response(),ChatCompletion.model_validate(response())):
        result=model._create_chat_result(raw)
        message=result.generations[0].message
        assert message.additional_kwargs['reasoning_content']=='Synthetic reasoning fragment.'
        assert message.tool_calls[0]['name']=='probe'
        assert message.usage_metadata['output_token_details']['reasoning']==3
        assert result.llm_output['provider_response']['id']=='test-response'
        attrs=chat_attributes({'messages':[]},{'responses':[message_record(message)]})
        assert 'Synthetic reasoning fragment.' in attrs['llm.output_messages.0.message.content']
        before=model._get_request_payload([message])
        clean=message.model_copy(deep=True);clean.additional_kwargs.pop('reasoning_content')
        assert before==model._get_request_payload([clean])


def test_stream_reasoning_concatenation_and_missing_field():
    model=CompatibleChatModel(model='test',api_key='offline-placeholder')
    chunks=[]
    for delta in ({'role':'assistant','reasoning_content':'First '},{'reasoning_content':'second.'},{'content':'Answer'}):
        chunks.append(model._convert_chunk_to_generation_chunk({'choices':[{'index':0,'delta':delta}]},AIMessageChunk,None))
    merged=chunks[0]+chunks[1]+chunks[2]
    assert merged.message.additional_kwargs['reasoning_content']=='First second.'
    assert merged.message.content=='Answer'
    raw=response();raw['choices'][0]['message'].pop('reasoning_content')
    assert 'reasoning_content' not in model._create_chat_result(raw).generations[0].message.additional_kwargs
