"""Check each API reference independently, without executing examples."""
import ast
import json
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, field_validator, BeforeValidator

class ResultSource(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tool_call_id: str = Field(description='真实来源工具调用引用；当前E编号由Harness恢复。未知引用仅使当前API不通过。')
    pointer: str = Field(default='', description='JSON字段路径，如/items/0/id；空串是整个结果。')

    @field_validator('tool_call_id')
    @classmethod
    def restore_reference(cls,value):
        from workers.evidence_refs import ACTIVE_EVIDENCE_REFS
        refs=ACTIVE_EVIDENCE_REFS.get() or {}
        return next((k for k,v in refs.items() if v==value),value)

class ApiParameter(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    required: bool
    purpose: str
    source: ResultSource | None = Field(default=None, description='参数值的真实来源。未取得填null，不猜值，不复制凭据。')

class ApiHandoffReceipt(BaseModel):
    model_config = ConfigDict(extra='forbid')
    api: 'ApiHandoff'
    validation_status: str = Field(pattern='^(VALIDATED|REJECTED)$')
    validation_error: str = Field(default='', max_length=300)


class ApiHandoff(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    documentation: ResultSource = Field(description='定位文档中的接口名称字段，值须与name完全相等。')
    parameters: list[ApiParameter] = Field(default_factory=list, max_length=40)
    call_example: str = Field(description='单个Python调用表达式，参数使用清晰变量占位或数值/布尔常量，不填秘密或猜测值。只展示不执行。')
    next_action: str = Field(default='', description='可选：接收者还需做什么。接口资料完整且无待办时留空，不强造下一步。')

ApiHandoffReceipt.model_rebuild()

def parse_api_list(value):
    from observability import trace_span,set_span_output
    accepted=[]
    for index,raw in enumerate(value if isinstance(value,list) else [value]):
        try: accepted.append(ApiHandoff.model_validate(raw))
        except (ValueError,TypeError):
            with trace_span('Handoff / Invalid API Entry',input_value={'index':index}) as span:
                set_span_output(span,{'status':'REJECTED','reason':'schema_invalid'})
    return accepted

ApiHandoffList = Annotated[list[ApiHandoff], BeforeValidator(parse_api_list)]

def api_field():
    return Field(default_factory=list,description='直接填写API列表，每项含name/purpose/documentation/parameters/call_example；next_action为可选待办。包含下一位Worker否则必须重新查找的全部已发现接口，尤其认证、读取、写入和回读接口。无接口交接填[]。Harness逐项标记已验证或失败原因，失败项不会静默删除；文件另走已有候选发布字段。')

def pointer_value(value, pointer):
    if not pointer: return value
    if not pointer.startswith('/'): raise ValueError('invalid pointer')
    for part in pointer[1:].split('/'):
        key=part.replace('~1','/').replace('~0','~')
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value

def decode_json_values(value):
    """Decode one JSON value or a stream produced by several print calls."""
    if not isinstance(value, str):
        return [value]
    decoder = json.JSONDecoder()
    values=[]
    index=0
    while index < len(value):
        while index < len(value) and value[index].isspace(): index += 1
        if index >= len(value): break
        decoded,index = decoder.raw_decode(value,index)
        values.append(decoded)
    return values

def document_identity(value):
    if isinstance(value,str): return value
    if not isinstance(value,dict): return None
    app=value.get('app_name')
    api=value.get('api_name')
    if isinstance(app,str) and isinstance(api,str): return f'{app}.{api}'
    name=value.get('name')
    return name if isinstance(name,str) else None

def safe_example_value(value):
    if isinstance(value,ast.Name): return True
    return isinstance(value,ast.Constant) and not isinstance(value.value,str)

def resolve_documentation(source, results, expected_name):
    record=results.get(source.tool_call_id)
    if record is None: raise ValueError('documentation source call missing')
    if record.get('status')=='error': raise ValueError('documentation source call failed')
    content=record.get('content')
    try:
        values=decode_json_values(content)
    except json.JSONDecodeError as error:
        raise ValueError('documentation source is not API documentation JSON') from error
    # Some tool adapters wrap the printed value once as {"content": "..."}.
    if len(values)==1 and isinstance(values[0],dict) and isinstance(values[0].get('content'),str):
        values=decode_json_values(values[0]['content'])
    if source.pointer:
        if len(values)!=1: raise ValueError('documentation pointer requires one JSON root')
        selected=pointer_value(values[0],source.pointer)
        if document_identity(selected)!=expected_name: raise ValueError('API name differs from documentation source')
        return selected
    matches=[value for value in values if document_identity(value)==expected_name]
    if len(matches)!=1:
        raise ValueError('expected exactly one matching API document')
    return matches[0]

def validate_apis(items, results):
    accepted=[]; errors=[]
    def resolve(source):
        record=results.get(source.tool_call_id)
        if record is None: raise ValueError('source call missing')
        if record.get('status')=='error': raise ValueError('source call failed')
        content=record.get('content')
        # An empty pointer cites the successful Tool result as a whole. Do not
        # force it through JSON: AppWorld legitimately returns several printed
        # values or plain text, while a pointer still requires structured data.
        if not source.pointer:
            if content is None or (isinstance(content,str) and not content.strip()):
                raise ValueError('source result is empty')
            if isinstance(content,dict) and (content.get('error') or content.get('status') in {'error','ERROR','FAILED'}):
                raise ValueError('source result reports failure')
            return content
        try:
            values=decode_json_values(content)
        except json.JSONDecodeError as error:
            raise ValueError('source pointer requires structured JSON') from error
        if len(values)==1 and isinstance(values[0],dict) and isinstance(values[0].get('content'),str):
            values=decode_json_values(values[0]['content'])
        if len(values)!=1:
            raise ValueError('source pointer requires one JSON root')
        data=values[0]
        if isinstance(data,dict) and (data.get('error') or data.get('status') in {'error','ERROR','FAILED'}):
            raise ValueError('source result reports failure')
        return pointer_value(data,source.pointer)
    for index,raw in enumerate(items):
        try:
            item=ApiHandoff.model_validate(raw)
            document=resolve_documentation(item.documentation,results,item.name)
            names=[p.name for p in item.parameters]
            if len(names)!=len(set(names)): raise ValueError('duplicate parameter')
            for p in item.parameters:
                if p.source is not None: resolve(p.source)
            expression=ast.parse(item.call_example,mode='eval').body
            called_name=ast.unparse(expression.func) if isinstance(expression,ast.Call) else ''
            canonical_called_name=called_name[5:] if called_name.startswith('apis.') else called_name
            if not isinstance(expression,ast.Call) or canonical_called_name!=item.name:
                raise ValueError('example API mismatch')
            documented_names={p.get('name') for p in document.get('parameters',[]) if isinstance(p,dict)} if isinstance(document,dict) else set()
            allowed_names=set(names)|{name for name in documented_names if isinstance(name,str)}
            if expression.args or any(k.arg is None or k.arg not in allowed_names or not safe_example_value(k.value) for k in expression.keywords):
                raise ValueError('example must use documented keywords and safe placeholders')
            if any(p.required and p.name not in {k.arg for k in expression.keywords} for p in item.parameters):
                raise ValueError('required example parameter missing')
            accepted.append(item)
        except (ValueError,TypeError,KeyError,IndexError,SyntaxError) as error:
            errors.append({'index':index,'name':raw.get('name') if isinstance(raw,dict) else None,
                           'status':'REJECTED','reason':type(error).__name__,'detail':str(error)[:240]})
    return accepted,errors
