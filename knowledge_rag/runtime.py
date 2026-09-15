"""Shared local retrieval infrastructure, separate memory/document admission."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import RLock
from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain.tools import ToolRuntime, tool
from observability import trace_span, set_span_output
from pydantic import Field

SHARED_CODE = ContextVar("shared_code_knowledge", default=None)


@contextmanager
def shared_code_scope(owner=None):
    token = SHARED_CODE.set(owner)
    try:
        yield
    finally:
        SHARED_CODE.reset(token)


ACTIVE = ContextVar("knowledge_scope", default=None)


class RetrievalHub:
    def __init__(self, memory, models, config=None):
        from knowledge_rag.service import PROJECT
        self.config = config or json.loads((PROJECT / "config/rag.json").read_text(encoding="utf-8"))
        self.root = PROJECT / self.config["root"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory, self.models = memory, models
        self.lock = RLock()
        self.embedding = None

    def _run(self, scope, operation):
        from knowledge_rag.service import KnowledgeBase, LocalEmbedding
        # One local model; Jina adapter changes and Qdrant file access are serial.
        with self.lock:
            if self.embedding is None:
                self.embedding = LocalEmbedding(self.config)
            kb = KnowledgeBase(scope, self.config, self.embedding)
            try:
                return operation(kb)
            finally:
                kb.close()

    async def sync(self, scope):
        from knowledge_rag.service import PROJECT, digest
        directories = [p.parent for p in (self.root / "uploads" / digest(scope)[:24]).rglob(".ready")]
        if scope == "owner":
            directories.append(PROJECT / self.config["inbox"])
        files = [p for d in directories for p in d.rglob("*") if p.is_file()]
        if not files:
            return []
        return await asyncio.to_thread(self._run, scope, lambda kb: [item for d in directories for item in kb.sync(d)])

    def snapshot(self, run, key, value=None):
        with closing(sqlite3.connect(self.root / "retrieval-state.sqlite3", timeout=20)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS snapshots (run TEXT, key TEXT, value TEXT, PRIMARY KEY(run,key))")
            if value is not None:
                db.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?,?)", (run,key,value))
            row = db.execute("SELECT value FROM snapshots WHERE run=? AND key=?", (run,key)).fetchone()
            return row[0] if row else None

    async def rag(self, scope, query):
        from knowledge_rag.service import digest
        inbox = self.root / "uploads" / digest(scope)[:24]
        # Empty libraries do not load GPU models or search unrelated data.
        if not any(inbox.rglob("*")) and not (self.root / digest(scope)[:24]).exists():
            return []
        def retrieve(kb):
            if self.models is None:
                raise RuntimeError("RAG relevance scorer unavailable")
            tokens = kb.embedding.tokenizer.encode(query, add_special_tokens=False)
            bounded_query = kb.embedding.tokenizer.decode(tokens[:self.config.get("query_tokens", 256)])
            with trace_span("RAG / Query Preparation", input_value={"original_query": query}) as span:
                set_span_output(span, {"search_query": bounded_query, "truncated": len(tokens) > self.config.get("query_tokens", 256)})
            return kb.search(bounded_query, scorer=lambda q, docs: self.models.rerank_windows(q, docs, top_k=len(docs), max_windows=self.config.get("rerank_windows", 2)))
        return await asyncio.to_thread(self._run, scope, retrieve)

    async def ingest(self, scope, paths):
        def read_for_ingest(path):
            """Parse a document and use isolated OCR only for unreadable PDFs."""
            from knowledge_rag.service import read_document

            path = Path(path)
            parse_error = None
            try:
                documents = read_document(path)
            except Exception as error:
                documents = []
                parse_error = error
            if any(document.page_content.strip() for document in documents):
                return documents
            if path.suffix.lower() != ".pdf":
                detail = f": {parse_error}" if parse_error else ""
                raise ValueError(f"{path.name}: 没有可提取文字{detail}")
            try:
                from tools.local_native import _reading_process

                result = _reading_process(path.read_bytes(), path.name, ocr="auto")
                markdown = str(result.get("markdown") or "").strip()
                if not markdown:
                    raise ValueError("OCR没有识别出文字")
                return [Document(page_content=markdown, metadata={
                    "parser": "ocr_markdown",
                    "ocr_summary": result.get("summary") or {},
                })]
            except Exception as error:
                original = f"；直接解析错误：{parse_error}" if parse_error else ""
                raise ValueError(
                    f"{path.name}: PDF没有可提取文字，OCR也未完成：{error}{original}"
                ) from error

        def ingest_all(kb):
            prepared = []
            for path in paths:
                documents = read_for_ingest(path)
                prepared.append((path, documents))
            return [kb.ingest(p, documents=documents) for p, documents in prepared]
        return await asyncio.to_thread(self._run, scope, ingest_all)

    def count_tokens(self, text):
        """Count with the same tokenizer used to budget retrieved context."""
        with self.lock:
            if self.embedding is None:
                return None
            return len(self.embedding.tokenizer.encode(text, add_special_tokens=False))


@contextmanager
def retrieval_scope(hub, scope, run):
    token = ACTIVE.set((hub, scope, run) if hub else None)
    try:
        yield
    finally:
        ACTIVE.reset(token)


async def automatic_rag(query, role, key=None, agent=None):
    active = ACTIVE.get()
    if not active or not query.strip():
        return ""
    hub, scope, run = active
    with trace_span(f"RAG / {role} Context", kind="retriever", input_value={"query": query, "automatic": True}) as span:
        try:
            snapshot_key = key or role
            existing = hub.snapshot(run, snapshot_key)
            if existing is not None:
                set_span_output(span, {
                    "status": "reused",
                    "injected_context": existing,
                    "serialized_context_tokens": hub.count_tokens(existing),
                })
                return existing
            results = await hub.rag(scope, query)
            content_tokens = sum(
                int(result.get("context_tokens") or 0)
                for result in results
            )
            if agent:
                from knowledge_rag.expansion import grants
                results = grants(hub,run,agent,results)
            else:
                from knowledge_rag.expansion import public_result
                results = [public_result(result) for result in results]
            from knowledge_rag.guidance import format_rag_context
            text = format_rag_context(results)
            set_span_output(span, {
                "results": results,
                "injected_context": text,
                "content_tokens": content_tokens,
                "serialized_context_tokens": hub.count_tokens(text),
                "memory_slots_used": 0,
            })
            hub.snapshot(run, snapshot_key, text)
            return text
        except Exception as error:
            set_span_output(span, {"status": "unavailable", "error": str(error), "injected_context": ""})
            return ""


@tool
async def search_knowledge(
    query: Annotated[str, Field(
        description="要查找的具体资料、接口名、字段名或错误信息；不要粘贴整段对话。",
        min_length=1,
        max_length=4000,
    )],
    runtime: ToolRuntime,
    source: Annotated[Literal["rag", "memory", "both"], Field(
        description="rag查文档/API；memory查用户偏好和历史事实；确实同时需要两类资料时才选both。",
    )] = "rag",
) -> dict:
    """搜索已经登记的资料。RAG只返回候选目录或章节；API目录需再用reference_id和精确api_name读取完整签名。"""
    active = ACTIVE.get()
    if not active:
        return {"status": "unavailable", "results": []}
    hub, scope, run = active
    agent = getattr(runtime,'state',{}).get('knowledge_agent_key') or str(runtime.config.get("configurable", {}).get("thread_id", "unknown"))
    with trace_span("Knowledge / Manual Search", kind="retriever", input_value={"query": query, "source": source, "agent": agent}) as span:
        result = {"status": "ok", "rag": [], "memory": ""}
        try:
            if not query.strip() or len(query) > 4000:
                raise ValueError("Query must contain 1–4000 characters")
            if source in {"rag", "both"}:
                result["rag"] = await hub.rag(scope, query)
                from knowledge_rag.expansion import grants
                result['rag'] = grants(hub,run,agent,result['rag'])
            if source in {"memory", "both"}:
                result["memory"] = hub.memory.format_context(await hub.memory.retrieve_for_turn(query))
        except Exception as error:
            result.update(status="unavailable", error=str(error))
        from knowledge_rag.guidance import search_hint
        result = search_hint(result)
        set_span_output(span, result)
        return result


def with_knowledge_tool(tools):
    from workers.history_archive import read_execution_history
    return [*tools, *[t for t in (search_knowledge,read_knowledge,read_execution_history) if not any(x.name==t.name for x in tools)]]


@tool
async def read_knowledge(
    reference_id: Annotated[str, Field(
        description="从自动RAG上下文或search_knowledge结果中原样复制的reference_id；禁止猜测或改写。",
        min_length=1,
        max_length=128,
    )],
    runtime: ToolRuntime,
    node_id: Annotated[str | None, Field(
        description="文档导航时从RAG目录条目原样复制的node_id。API目录优先使用api_name；省略两者时读取最初命中的父节点。",
        max_length=128,
    )] = None,
    api_name: Annotated[str | None, Field(
        description="读取API完整签名时，从同一reference_id的api_names中原样复制精确名称；不得猜测。不能与node_id同时填写。",
        min_length=1,
        max_length=200,
    )] = None,
    view: Annotated[Literal['content','children'], Field(
        description="content读取节点原文或完整API签名；children只查看节点下的目录。",
    )] = 'content',
    offset: Annotated[int, Field(
        description="续页位置。首次读取填0；后续必须原样复制上次返回的next_offset。",
        ge=0,
    )] = 0,
    depth: Annotated[int, Field(
        description="children模式向下展示的目录层数；content模式忽略此值。",
        ge=1,
        le=8,
    )] = 2,
) -> dict:
    """按需读取RAG候选。API目录用reference_id+api_name读取完整签名；普通文档用node_id导航；未结束时按next_offset续页。"""
    active=ACTIVE.get()
    if not active:return {'status':'unavailable'}
    hub,scope,run=active
    agent=getattr(runtime,'state',{}).get('knowledge_agent_key') or str(runtime.config.get('configurable',{}).get('thread_id','unknown'))
    from knowledge_rag.expansion import grants,read_page
    with trace_span('RAG / Browse Children' if view=='children' else 'RAG / Read Node',kind='retriever',input_value={'reference_id':reference_id,'offset':offset,'agent':agent,'view':view,'node_id':node_id,'api_name':api_name,'depth':depth}) as span:
        grant=grants(hub,run,agent).get(reference_id)
        if not grant:result={'status':'invalid_reference'}
        else:
            try:
                result=await asyncio.to_thread(hub._run,scope,lambda kb:read_page(kb,grant,offset,view=view,node_id=node_id,api_name=api_name,depth=depth))
            except Exception as error:result={'status':'unavailable','error':str(error)}
        from knowledge_rag.guidance import read_hint
        result = read_hint(result, reference_id=reference_id, offset=offset,
                           view=view, node_id=node_id, api_name=api_name, depth=depth)
        set_span_output(span,result)
        return result
