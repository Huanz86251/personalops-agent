"""Automatic role-local retrieval and conditional original-document tool."""
import asyncio
from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain.agents import AgentState
from knowledge_rag.runtime import ACTIVE, SHARED_CODE, automatic_rag
from knowledge_rag.query import retrieval_query as task_query
from knowledge_rag.expansion import grants


class KnowledgeState(AgentState):
    knowledge_agent_key: str
    knowledge_context: str


class KnowledgeMiddleware(AgentMiddleware[KnowledgeState]):
    state_schema=KnowledgeState
    def __init__(self,role): self.role=role

    async def abefore_agent(self,state,runtime,config: RunnableConfig):
        if not ACTIVE.get():return None
        agent=str(config.get('configurable',{}).get('thread_id','unknown'))+':'+self.role
        if state.get('knowledge_agent_key')==agent:return None
        messages=state.get('messages',[])
        human=[m for m in messages if (m.get('role') if isinstance(m,dict) else m.type) in {'user','human'} and not (m.get('additional_kwargs',{}) if isinstance(m,dict) else m.additional_kwargs).get('personalops_runtime_event')]
        assignment=(human[-1].get('content') if isinstance(human[-1],dict) else human[-1].content) if human else ''
        query=task_query(assignment,state.get('code_task'))
        owner=SHARED_CODE.get()
        if owner is not None:
            from observability import trace_span,set_span_output
            hub,scope,run=ACTIVE.get()
            with trace_span('RAG / Share Task Context',input_value={'owner':owner,'recipient':agent}) as span:
                results=grants(hub,run,agent,list(grants(hub,run,owner).values()))
                from knowledge_rag.guidance import format_rag_context
                text=format_rag_context(results)
                set_span_output(span,{'retrieval_performed':False,'results':results,'injected_context':text})
        else:
            text=await automatic_rag(query,self.role,key=agent,agent=agent)
        update={'knowledge_agent_key':agent,'knowledge_context':text}
        if text:update['messages']=[HumanMessage(content=text,additional_kwargs={
            'personalops_runtime_event':True, 'knowledge_source':True})]
        return update

    def before_agent(self,state,runtime,config: RunnableConfig):
        return asyncio.run(self.abefore_agent(state,runtime,config))

    def filtered(self,request):
        active=ACTIVE.get(); agent=request.state.get('knowledge_agent_key')
        allowed=bool(active and agent and grants(active[0],active[2],agent))
        gated={'read_knowledge','search_knowledge'}
        tools=[t for t in request.tools if (getattr(t,'name',None) or (t.get('name') if isinstance(t,dict) else None)) not in gated or allowed]
        return request.override(tools=tools)

    def wrap_model_call(self,request,handler):return handler(self.filtered(request))
    async def awrap_model_call(self,request,handler):return await handler(self.filtered(request))
