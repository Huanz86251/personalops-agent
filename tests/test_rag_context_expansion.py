"""Ranking scores compact children before selected document parents are returned."""
from types import SimpleNamespace
from langchain_core.documents import Document
from test_knowledge_rag import RagTests


class ContextTests(RagTests):
    def test_dynamic_limit_expands_only_near_tied_distinct_parents(self):
        kb=self.open()
        kb.config.update(automatic_catalog_limit=2,dynamic_catalog_max_limit=4,dynamic_catalog_score_margin=.01)
        docs=[Document(page_content=n,metadata={"source_id":"s","parent_id":n,"retrieval_kind":"document_section"}) for n in ("first","second","third","fourth","fifth")]
        scores={"first":.60,"second":.55,"third":.545,"fourth":.541,"fifth":.50}
        try:
            limit,policy=kb._dynamic_result_limit(docs,scores)
            self.assertEqual(limit,4)
            self.assertEqual(policy["hard_max_limit"],4)
            scores["third"]=.53
            limit,policy=kb._dynamic_result_limit(docs,scores)
            self.assertEqual(limit,2)
            self.assertEqual(policy["boundary_score"],.55)
        finally:
            kb.close()

    def test_reranker_changes_order_before_expansion(self):
        kb = self.open()
        file = self.base / 'guide.md'
        file.write_text('# API\n\n## SUM\nSUM adds numbers.\n\n## Login\nLogin requires credentials.', encoding='utf-8')
        try:
            kb.ingest(file)
            captured = []
            def scorer(query, docs):
                captured.extend(docs)
                return [SimpleNamespace(index=i, score=.9 if '/ Login\n' in text else .5) for i,text in enumerate(docs)]
            hits = kb.search('SUM', scorer=scorer)
            self.assertIn('/ Login', hits[0]['path'])
            self.assertEqual(len(hits), 2)
            self.assertGreater(hits[0]['relevance_score'], hits[1]['relevance_score'])
            self.assertTrue(all(len(h['text']) <= 400 for h in hits))
            self.assertEqual(len(captured), 2)
        finally: kb.close()

    def test_chunks_link_neighbors_and_do_not_cross_sections(self):
        kb = self.open()
        try:
            parents, chunks = kb.chunk([Document(page_content='# A\n\n```\n'+('code example '*50)+'\n```\n\n# B\n\nOther section.')], 'source', 'guide.md')
            by_id={c.metadata['chunk_id']:c for c in chunks}
            self.assertTrue(any(c.metadata['previous_chunk_id'] for c in chunks))
            for child in chunks:
                self.assertIn(child.metadata['parent_id'], parents)
                for field in ('previous_chunk_id','next_chunk_id'):
                    target=child.metadata[field]
                    if target:self.assertEqual(by_id[target].metadata['parent_id'],child.metadata['parent_id'])
        finally: kb.close()
