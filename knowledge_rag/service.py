"""LangChain chunking/indexing and Qdrant retrieval with scoped local storage."""
from __future__ import annotations

import hashlib
import json
import re
from uuid import UUID
from pathlib import Path

from filelock import FileLock
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index
from langchain_classic.indexes import SQLRecordManager
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from observability import trace_span, set_span_output

PROJECT = Path(__file__).resolve().parents[1]
SUPPORTED = {".txt", ".md", ".json", ".csv", ".html", ".pdf", ".docx", ".xlsx", ".pptx", ".jsonl"}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def terms(text):
    import jieba
    # Keep API identifiers intact; tokenize Chinese separately.
    return [t.casefold() for t in jieba.lcut(text) if re.search(r"\w", t)]


class MatchingBM25Retriever(BM25Retriever):
    """Do not let arbitrary zero-match rankings pollute bilingual rank fusion."""
    def _get_relevant_documents(self, query, *, run_manager):
        candidates = super()._get_relevant_documents(query, run_manager=run_manager)
        query_terms = set(self.preprocess_func(query))
        return [doc for doc in candidates if query_terms.intersection(self.preprocess_func(doc.page_content))]


class LocalEmbedding(Embeddings):
    def __init__(self, config):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config["model"], device=config["device"],
            revision=config.get("revision"), trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16} if config["device"] == "cuda" else {})
        self.model.max_seq_length = config["embedding_max_tokens"]
        self.tokenizer = self.model.tokenizer
        self.max_tokens = config["embedding_max_tokens"]
        self.batch = config["batch_size"]
        self.dimensions = config["dimensions"]
        self.v5 = "jina-embeddings-v5" in config["model"]
        native = self.model.get_embedding_dimension()
        if native is None:
            native = getattr(getattr(self.model[0], "config", None), "hidden_size", None)
        if native is None:
            raise ValueError("Model did not expose its native embedding dimension")
        if native != self.dimensions and not (self.v5 and self.dimensions in {32,64,128,256,512,768,1024} and self.dimensions <= native):
            raise ValueError("Embedding dimensions differ from the configured index.")

    def _encode(self, texts, mode):
        if any(len(self.tokenizer.encode(t)) > self.max_tokens - 32 for t in texts):
            raise ValueError("Embedding input too long; explicit splitting is required")
        options = {}
        if self.v5:
            options = {"task": "text-matching" if mode == "semantic" else "retrieval",
                       "truncate_dim": self.dimensions}
            if mode != "semantic":
                options["prompt_name"] = "query" if mode == "query" else "document"
            else:
                options["prompt"] = ""  # Suppress the model's default document prompt.
        return self.model.encode(texts, batch_size=self.batch,
                                 normalize_embeddings=True, **options).tolist()

    def embed_documents(self, texts):
        return self._encode(texts, "document")

    def embed_query(self, text):
        return self._encode([text], "query")[0]

    def semantic_embeddings(self):
        owner = self
        class MatchingEmbedding(Embeddings):
            def embed_documents(self, texts):
                return owner._encode(texts, "semantic")
            def embed_query(self, text):
                return self.embed_documents([text])[0]
        return MatchingEmbedding()


def read_document(path):
    from knowledge_rag.structured import parse
    return parse(path)


class KnowledgeBase:
    """One scope per storage directory; callers must supply their trusted scope."""
    def __init__(self, scope="local", config=None, embedding=None, token_count=None):
        self.config = config or json.loads((PROJECT / "config/rag.json").read_text(encoding="utf-8"))
        self.root = PROJECT / self.config["root"] / digest(scope)[:24]
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.root / "writer.lock"))
        self.lock.acquire(timeout=1)
        self.client = None
        try:
            self.embedding = embedding or LocalEmbedding(self.config)
            self.count = token_count or (lambda text: len(self.embedding.tokenizer.encode(text, add_special_tokens=False)))
            self.fingerprint = digest(json.dumps({k:v for k,v in self.config.items() if k not in {
                "rerank_windows", "query_facets", "dynamic_catalog_max_limit",
                "dynamic_catalog_score_margin",
            }}, sort_keys=True))
            self.client = QdrantClient(path=str(self.root / "vectors"))
            self.collection = "documents_" + self.fingerprint[:16]
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                    size=self.config["dimensions"], distance=models.Distance.COSINE))
            self.store = QdrantVectorStore(client=self.client, collection_name=self.collection,
                                           embedding=self.embedding, validate_embeddings=False)
            self.records = SQLRecordManager(self.collection, db_url="sqlite:///" + str(self.root / "records.sqlite3"))
            self.records.create_schema()
            self.manifest_path = self.root / (self.collection + ".json")
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
        except Exception:
            self.close()
            raise

    def close(self):
        if self.client is not None:
            self.client.close()
        if getattr(self, "records", None) is not None:
            self.records.engine.dispose()
        self.lock.release()

    def _splitter(self, size, overlap=0):
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
            length_function=self.count, separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""])

    def chunk(self, documents, source_id, filename):
        from knowledge_rag.structured import chunk_documents
        return chunk_documents(self, documents, source_id, filename)

    def ingest(self, path, source_id=None, *, documents=None):
        path = Path(path).resolve()
        source_id = source_id or str(path)
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.manifest.get(source_id, {}).get("hash") == content_hash:
            return {"status": "unchanged", "source": path.name}
        with trace_span("RAG / Index Document") as span:
            documents = read_document(path) if documents is None else documents
            if not any(d.page_content.strip() for d in documents):
                raise ValueError("No readable text; OCR may be required. Previous index retained.")
            with trace_span("RAG / Chunk") as chunk_span:
                parents, children = self.chunk(documents, source_id, path.name)
                set_span_output(chunk_span, {"parents": len(parents), "chunks": len(children),
                    "structure": parents, "chunk_details": [{"metadata": c.metadata,
                    "tokens": self.count(c.page_content), "text": c.page_content} for c in children]})
            if not children:
                raise ValueError("No chunks produced")
            blobs = self.root / 'node-content'
            blobs.mkdir(exist_ok=True)
            for parent in parents.values():
                if 'content' in parent:
                    content = parent.pop('content')
                    ref = digest(content)
                    target = blobs / (ref+'.txt')
                    if not target.exists(): target.write_text(content,encoding='utf-8')
                    parent['content_ref'] = ref
            result = index(children, self.records, self.store, cleanup="incremental", source_id_key="source_id",
                key_encoder=lambda doc: str(UUID(hex=digest(json.dumps(
                    {"text": doc.page_content, "metadata": doc.metadata}, sort_keys=True, ensure_ascii=False))[:32])))
            self.manifest[source_id] = {"hash": content_hash, "parents": parents}
            temp = self.manifest_path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")
            temp.replace(self.manifest_path)
            result = {"status": "indexed", "source": path.name, "chunks": len(children), **result}
            set_span_output(span, result)
            return result

    def sync(self, directory):
        results = []
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED:
                try:
                    results.append(self.ingest(path))
                except Exception as error:
                    results.append({"source": path.name, "status": "error", "error": str(error)})
        return results

    def search(self, query, scorer=None):
        from knowledge_rag.structured import display_path
        def root_of(nodes, ident):
            while nodes[ident].get('parent_id'):
                ident = nodes[ident]['parent_id']
            return ident
        with trace_span("RAG / Retrieve") as span:
            points, offset = [], None
            while True:
                batch, offset = self.client.scroll(self.collection, limit=256, offset=offset, with_vectors=False)
                points.extend(batch)
                if offset is None:
                    break
            docs = [Document(page_content=p.payload["page_content"], metadata=p.payload["metadata"]) for p in points]
            if not docs:
                return []
            # An explicitly named, indexed API is the only automatic leaf read.
            # Natural-language matches are promoted to a sibling catalog below.
            exact = []
            for doc in docs:
                api_name = doc.metadata.get('api_name')
                if api_name and re.search(r'(?<![A-Za-z0-9_])' + re.escape(api_name) + r'(?![A-Za-z0-9_])', query, re.I):
                    exact.append(doc)
            if exact:
                results, exact_seen = [], set()
                for hit in exact[:self.config.get('exact_api_limit', 4)]:
                    source_id = hit.metadata['source_id']
                    node_id = hit.metadata['leaf_node_id']
                    if (source_id, node_id) in exact_seen:
                        continue
                    exact_seen.add((source_id, node_id))
                    node = self.manifest[source_id]['parents'][node_id]
                    text = (self.root / 'node-content' / (node['content_ref'] + '.txt')).read_text(encoding='utf-8')
                    limit = self.config.get('retrieval_chunk_tokens', 1200)
                    text = self._fit_text(text, limit)
                    results.append({
                        'kind': 'api_endpoint', 'match': 'exact_api_name',
                        'source': hit.metadata['source'], 'node_id': node_id,
                        'parent_id': node_id, 'path': display_path(node['path'], self),
                        'root_node_id': root_of(self.manifest[source_id]['parents'], node_id),
                        'api_name': hit.metadata['api_name'], 'text': text,
                        'context_tokens': self.count(text), 'relevance_score': 1.0,
                        '_source_id': source_id, '_version': self.manifest[source_id]['hash'],
                        '_collection': self.collection,
                    })
                set_span_output(span, {'query': query, 'mode': 'exact_api_name', 'results': results,
                                       'memory_slots_used': 0})
                return results
            k = self.config["candidate_limit"]
            bm25 = MatchingBM25Retriever.from_documents(docs, preprocess_func=terms, k=k)
            dense = self.store.as_retriever(search_kwargs={"k": k})
            with trace_span("RAG / Hybrid Candidates", kind="retriever", input_value={"query": query, "per_route_limit": k}) as candidates_span:
                dense_pairs = self.store.similarity_search_with_score(query, k=k)
                lexical_hits = bm25.invoke(query)
                fusion = EnsembleRetriever(retrievers=[dense, bm25], weights=[self.config.get("dense_weight", 0.5), self.config.get("bm25_weight", 0.5)], c=self.config.get("rrf_constant", 60))
                combined = fusion.weighted_reciprocal_rank([[d for d, _ in dense_pairs], lexical_hits])
                set_span_output(candidates_span, {
                    "dense": [{"rank": i+1, "cosine_score": float(score), "text": d.page_content} for i, (d, score) in enumerate(dense_pairs)],
                    "bm25": [{"rank": i+1, "text": d.page_content} for i, d in enumerate(lexical_hits)],
                    "bm25_zero_match_votes": 0, "weights": fusion.weights, "rrf_constant": fusion.c,
                    "fused": [{"rank": i+1, "text": d.page_content} for i, d in enumerate(combined)],
                })
            ordered = list({(d.metadata["parent_id"], d.page_content): d for d in combined}.values())[:self.config.get("gate_candidate_limit", 24)]
            score_by_text = {}
            audit_candidates = ordered[:]
            candidate_scores = {}
            if scorer is not None and ordered:
                with trace_span("RAG / Rerank and Filter", kind="retriever", input_value={"query": query, "threshold": self.config.get("relevance_threshold", 0.4), "candidates": [d.page_content for d in ordered]}) as gate_span:
                    scored = scorer(query, [d.page_content for d in ordered])
                    scores = {item.index: float(item.score) for item in scored}
                    candidate_scores = scores
                    score_by_text = {d.page_content: scores.get(i) for i, d in enumerate(ordered)}
                    accepted = [(i, d) for i, d in enumerate(ordered) if scores.get(i, -1) >= self.config.get("relevance_threshold", 0.4)]
                    accepted.sort(key=lambda item: -scores[item[0]])
                    set_span_output(gate_span, {"scores": scores, "accepted_indices": [i for i, _ in accepted],
                        "ranking_changed": [i for i,_ in accepted] != sorted(i for i,_ in accepted),
                        "scored_expanded_context": False,
                        "score_meaning":"Cross Encoder relevance; not memory confidence or correctness probability",
                        "candidates":[{'candidate_index':i,'chunk_id':d.metadata.get('chunk_id'),
                            'source':d.metadata.get('source'),'path':d.metadata.get('full_path'),
                            'text':d.page_content,'cross_encoder_score':scores.get(i),
                            'threshold_passed':scores.get(i,-1)>=self.config.get('relevance_threshold',.4)}
                            for i,d in enumerate(ordered)]})
                ordered = [d for _, d in accepted]
            result_limit, selection_policy = self._dynamic_result_limit(ordered, score_by_text)
            results, seen, used = [], set(), 0
            for hit in ordered:
                source_id = hit.metadata['source_id']
                nodes = self.manifest[source_id]['parents']
                is_api = hit.metadata.get('retrieval_kind') == 'api_endpoint'
                parent_id = (hit.metadata.get('catalog_parent_id') if is_api
                             else hit.metadata.get('parent_id'))
                selection_id = (source_id, parent_id)
                if selection_id in seen:
                    continue
                limit = min(self.config.get("retrieval_chunk_tokens", 1200),
                            self.config["automatic_context_tokens"] - used)
                if limit <= 0:
                    break
                if is_api:
                    node = nodes[parent_id]
                    all_entries = list(node.get('entries') or [])
                    matched = [candidate for candidate in ordered
                               if candidate.metadata.get('catalog_parent_id') == parent_id
                               and candidate.metadata.get('source_id') == source_id]
                    # Keep the global Cross Encoder order inside the selected
                    # parent. Alphabetical sorting used to erase that leaf rank.
                    matched_names = []
                    for candidate in matched:
                        api_name = candidate.metadata.get('api_name')
                        if api_name and api_name not in matched_names:
                            matched_names.append(api_name)
                    by_name = {entry['api_name']: entry for entry in all_entries}
                    ranked_entries = [
                        by_name[name] for name in matched_names if name in by_name
                    ]
                    entries = []
                    for entry in ranked_entries[:self.config.get('catalog_detail_limit', 4)]:
                        rendered = '\n'.join(
                            f"{item['relative_path']} | {item['method']} {item['path']} | {item['description']}"
                            for item in [*entries, entry]
                        )
                        if self.count(rendered) > limit:
                            break
                        entries.append(entry)
                    text = '\n'.join(
                        f"{item['relative_path']} | {item['method']} {item['path']} | {item['description']}"
                        for item in entries
                    )
                    text = self._fit_text(text, limit)
                    result = {
                        'kind': 'api_catalog', 'source': hit.metadata['source'],
                        'node_id': parent_id, 'parent_id': parent_id,
                        'root_node_id': root_of(nodes, parent_id),
                        'path': display_path(node['path'], self),
                        'matched_api_names': matched_names,
                        'api_names': [entry['api_name'] for entry in all_entries],
                        'catalog_only': True,
                        'entries': entries, 'text': text,
                        'relevance_score': max(score_by_text.get(candidate.page_content) or 0 for candidate in matched),
                        'context_tokens': self.count(text), '_source_id': source_id,
                        '_version': self.manifest[source_id]['hash'], '_collection': self.collection,
                    }
                    neighbors = []
                else:
                    node = nodes[parent_id]
                    raw = ''
                    if node.get('content_ref'):
                        raw = (self.root / 'node-content' / (node['content_ref'] + '.txt')).read_text(encoding='utf-8')
                    text = self._fit_text(raw or hit.page_content, limit)
                    result = {
                        'kind': 'document_section', 'source': hit.metadata['source'],
                        'node_id': parent_id, 'parent_id': parent_id,
                        'root_node_id': root_of(nodes, parent_id),
                        'path': display_path(node['path'], self), 'text': text,
                        'matched_path': display_path(hit.metadata.get('full_path') or node['path'], self),
                        'relevance_score': score_by_text.get(hit.page_content),
                        'context_tokens': self.count(text), '_source_id': source_id,
                        '_version': self.manifest[source_id]['hash'], '_collection': self.collection,
                    }
                    neighbors = []
                with trace_span("RAG / Promote Selected Parent", input_value={
                        "chunk_id": hit.metadata.get("chunk_id"), "parent_id": parent_id,
                        "scored_text": hit.page_content, "token_limit": limit}) as expand_span:
                    set_span_output(expand_span, {"kind": result['kind'], "text": text,
                        "tokens": self.count(text), "neighbor_ids": neighbors,
                        "leaf_returned": False})
                if not text:
                    continue
                seen.add(selection_id); used += self.count(text)
                results.append(result)
                if len(results) >= result_limit:
                    break
            selected_ids={(r['_source_id'],r['parent_id']):i+1 for i,r in enumerate(results)}
            ranked_ids={d.metadata.get('chunk_id'):i+1 for i,d in enumerate(ordered)}
            decisions=[]
            for i,d in enumerate(audit_candidates):
                ident=d.metadata.get('chunk_id')
                promoted=d.metadata.get('catalog_parent_id') or d.metadata.get('parent_id')
                selection_key=(d.metadata.get('source_id'), promoted)
                reason='selected_parent' if selection_key in selected_ids else ('missing_score' if scorer is not None and i not in candidate_scores else
                    'below_threshold' if scorer is not None and candidate_scores[i]<self.config.get('relevance_threshold',.4) else
                    'same_parent_already_selected' if (d.metadata.get('source_id'), promoted) in seen else
                    'outside_dynamic_limit' if len(selected_ids) >= result_limit else 'context_limit')
                decisions.append({'chunk_id':ident,'candidate_index':i,'source':d.metadata.get('source'),
                    'cross_encoder_score':candidate_scores.get(i),'rank_after_filter':ranked_ids.get(ident),
                    'promoted_parent_id': promoted, 'selected_rank':selected_ids.get(selection_key),'decision':reason})
            set_span_output(span, {"query": query, "context_tokens": used, "results": results,
                "selection_policy": selection_policy,
                "selection_decisions":decisions,"memory_slots_used":0})
            return results

    def _dynamic_result_limit(self, ordered, score_by_text):
        """Expand near-tied, distinct promoted parents without widening every query."""
        base = max(1, int(self.config.get("automatic_catalog_limit", 2)))
        hard_max = max(base, int(self.config.get("dynamic_catalog_max_limit", base)))
        margin = max(0.0, float(self.config.get("dynamic_catalog_score_margin", 0.01)))
        parent_scores, seen = [], set()
        for hit in ordered:
            source_id = hit.metadata.get("source_id")
            is_api = hit.metadata.get("retrieval_kind") == "api_endpoint"
            parent_id = hit.metadata.get("catalog_parent_id") if is_api else hit.metadata.get("parent_id")
            identity = (source_id, parent_id)
            if identity in seen: continue
            seen.add(identity)
            parent_scores.append(score_by_text.get(hit.page_content))
        limit = min(base, len(parent_scores))
        boundary_score = parent_scores[limit - 1] if limit else None
        if boundary_score is not None:
            for score in parent_scores[limit:hard_max]:
                if score is None or boundary_score - score > margin: break
                limit += 1
        return limit, {"mode":"near_tie_parent_expansion","base_limit":base,
            "hard_max_limit":hard_max,"score_margin":margin,"boundary_score":boundary_score,
            "selected_limit":limit,"distinct_parent_scores":parent_scores[:hard_max]}

    def _fit_text(self, text, limit):
        # Character trimming checked by the configured embedding tokenizer.
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo+hi+1)//2
            if self.count(text[:mid]) <= limit: lo = mid
            else: hi = mid-1
        return text[:lo]
