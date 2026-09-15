"""Offline engineering tests: synthetic embeddings do not measure model quality."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from langchain_core.embeddings import Embeddings
from knowledge_rag.service import KnowledgeBase, PROJECT


class SyntheticEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, float('SUM' in t), float('邮件' in t), .1] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class RagTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.config = json.loads((PROJECT/'config/rag.json').read_text())
        # Keep the legacy-layout regressions independent of production defaults.
        self.config.pop('reading_block_tokens', None)
        self.config.pop('structure_completion_tokens', None)
        self.config['retrieval_chunk_tokens'] = 400
        self.config.update(root=str(self.base/'index'), dimensions=4,
                           child_tokens=80, overlap_tokens=12, parent_tokens=250,
                           embedding_max_tokens=1024, automatic_context_tokens=500)
        self.trace = patch('observability._TRACER', None)
        # Avoid exporting fixture traces to the user's project.
        self.setup_trace = patch('observability.setup_observability')
        self.setup_trace.start(); self.trace.start()

    def tearDown(self):
        self.trace.stop(); self.setup_trace.stop(); self.directory.cleanup()

    def open(self, scope='test'):
        return KnowledgeBase(scope, self.config, SyntheticEmbedding(), len)

    def test_bm25_no_cross_language_match_has_no_arbitrary_votes(self):
        from knowledge_rag.service import MatchingBM25Retriever, terms
        retriever = MatchingBM25Retriever.from_texts(['SUM adds numbers.', 'Login refreshes credentials.'], preprocess_func=terms)
        self.assertEqual(retriever.invoke('数字求和？'), [])
        self.assertEqual(len(retriever.invoke('SUM')), 1)

    def test_v5_uses_distinct_retrieval_and_semantic_tasks(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import Mock
        from knowledge_rag.service import LocalEmbedding
        model = object.__new__(LocalEmbedding)
        model.model = SimpleNamespace(encode=Mock(return_value=np.zeros((1,512))))
        model.tokenizer = SimpleNamespace(encode=lambda text:list(text))
        model.max_tokens=1024; model.batch=8; model.dimensions=512; model.v5=True
        model.embed_documents(['document'])
        self.assertEqual(model.model.encode.call_args.kwargs['prompt_name'],'document')
        model.embed_query('query')
        self.assertEqual(model.model.encode.call_args.kwargs['prompt_name'],'query')
        model.semantic_embeddings().embed_documents(['sentence'])
        self.assertEqual(model.model.encode.call_args.kwargs['task'],'text-matching')
        self.assertNotIn('prompt_name',model.model.encode.call_args.kwargs)
        self.assertEqual(model.model.encode.call_args.kwargs['prompt'],'')
        with self.assertRaises(ValueError):model.embed_query('x'*1024)

    def test_incremental_update_scope_and_exact_parent(self):
        source=self.base/'formulas.md'
        source.write_text('# Functions\n\n## SUM\n\nSUM adds numbers. 参数是数字。\n\n## SUMMARY\n\nSUMMARY summarizes documents.',encoding='utf-8')
        kb=self.open()
        try:
            self.assertEqual(kb.ingest(source)['status'],'indexed')
            self.assertEqual(kb.ingest(source)['status'],'unchanged')
            hits=kb.search('SUM')
            self.assertIn('/ SUM\n',hits[0]['text'])
            self.assertLessEqual(sum(len(h['text']) for h in hits),500)
            source.write_text('# New\n\nReplacement content.',encoding='utf-8')
            result=kb.ingest(source)
            self.assertGreater(result['num_deleted'],0)
            self.assertFalse(any('/ SUM\n' in h['text'] for h in kb.search('SUM')))
        finally: kb.close()
        other=self.open('other')
        try:self.assertEqual(other.search('SUM'),[])
        finally:other.close()

    def test_empty_document_keeps_previous_and_reopen_skips(self):
        source=self.base/'manual.txt'; source.write_text('Useful original content.',encoding='utf-8')
        kb=self.open()
        try:kb.ingest(source)
        finally:kb.close()
        kb=self.open()
        try:
            self.assertEqual(kb.ingest(source)['status'],'unchanged')
            source.write_text('',encoding='utf-8')
            with self.assertRaises(ValueError):kb.ingest(source)
            self.assertTrue(kb.search('original'))
        finally:kb.close()

    def test_large_chinese_document_has_bounded_children(self):
        from langchain_core.documents import Document
        kb=self.open()
        try:
            parents,children=kb.chunk([Document(page_content='# 使用说明\n\n'+('这是一段登录说明。请保持访问凭据。' * 80))],'source','manual.md')
            self.assertGreater(len(children),1)
            self.assertTrue(all(len(c.page_content)<=1024 for c in children))
            self.assertTrue(all(c.metadata['parent_id'] in parents for c in children))
        finally:kb.close()

if __name__=='__main__':unittest.main()
