"""Build retrieval queries from task semantics, never executor bookkeeping."""
import json
import re


def task_query(assignment, contract=None, *, use_rag_query=False):
    text = assignment if isinstance(assignment,str) else str(assignment or '')
    data = {}
    # Existing Step envelopes carry JSON, followed by reports and budgets.
    match = re.search(r'当前Step[^\n]*：\s*', text)
    if match:
        try:
            step,_ = json.JSONDecoder().raw_decode(text[match.end():])
            explicit = step.get('rag_query')
            if use_rag_query and isinstance(explicit, str) and explicit.strip():
                return explicit.strip()
            for key in ('objective','success_criteria'):
                if step.get(key): data[key]=step[key]
            contract = contract or step.get('code_task')
        except (ValueError,TypeError):
            pass
    if hasattr(contract,'model_dump'): contract=contract.model_dump()
    if isinstance(contract,dict):
        for key in ('requirements','interfaces','validation_expectations'):
            if contract.get(key): data[key]=contract[key]
    if not data:
        # Plain user tasks remain intact; known generated envelope sections do not.
        match=re.search(r'用户原始请求：\s*(.*?)(?=\n整体目标：|\n当前Step|\n本次Attempt预算：|$)',text,re.S)
        if match: text=match.group(1).strip()
        data['task']=text
    return json.dumps(data,ensure_ascii=False,default=str)


def retrieval_query(assignment, contract=None):
    """Only the Scheduler's explicit query; no task-envelope fallback."""
    text = assignment if isinstance(assignment, str) else ''
    match = re.search(r'当前Step[^\n]*：\s*', text)
    if not match:
        return ''
    try:
        step, _ = json.JSONDecoder().raw_decode(text[match.end():])
        value = step.get('rag_query')
        return value.strip() if isinstance(value, str) else ''
    except (ValueError, TypeError, AttributeError):
        return ''
