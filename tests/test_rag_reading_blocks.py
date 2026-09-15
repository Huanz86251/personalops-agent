"""Reading-block boundary and short-window regressions; no model requests."""
import json
from types import SimpleNamespace
from langchain_core.documents import Document
from test_knowledge_rag import RagTests
from knowledge_rag.service import read_document


class ReadingBlockTests(RagTests):
    def setUp(self):
        super().setUp()
        self.config.update(reading_block_tokens=900, structure_completion_tokens=200,
                           retrieval_chunk_tokens=1200, automatic_context_tokens=2400)

    def test_json_completion_and_short_windows(self):
        kb = self.open()
        try:
            value = {'name': 'example', 'description': 'x' * 980}
            parents, chunks = kb.chunk([Document(page_content=json.dumps(value), metadata={'parser':'json'})], 's', 'api.json')
            self.assertGreater(len(chunks), 1)
            self.assertEqual(len({c.metadata['reading_block_id'] for c in chunks}), 1)
            for c in chunks:
                self.assertEqual(json.loads(c.metadata['reading_body']), value)
                self.assertGreater(c.metadata['completion_tokens'], 0)
                self.assertLessEqual(c.metadata['completion_tokens'], 200)
                self.assertLessEqual(len(c.metadata['body']), 80)
        finally:
            kb.close()

    def test_separate_records_not_packed(self):
        kb = self.open()
        try:
            values = [{'name':'A'}, {'name':'B'}]
            _, chunks = kb.chunk([Document(page_content=json.dumps(values),metadata={'parser':'json'})], 's', 'api.json')
            self.assertEqual(len({c.metadata['reading_block_id'] for c in chunks}), 2)
        finally:
            kb.close()

    def test_search_scores_windows_returns_distinct_full_blocks(self):
        kb = self.open()
        try:
            p = self.base / 'apis.json'
            values = [{'name': name, 'description': 'SUM example ' * 60} for name in ('alpha', 'beta')]
            p.write_text(json.dumps(values), encoding='utf-8')
            kb.ingest(p)
            scored = []
            def scorer(query, texts):
                scored.extend(texts)
                return [SimpleNamespace(index=i, score=.9-i*.001) for i in range(len(texts))]
            hits = kb.search('SUM', scorer=scorer)
            self.assertEqual(len(hits), 2)
            restored = [json.loads(h['text']) for h in hits]
            self.assertEqual({r['name'] for r in restored}, {'alpha', 'beta'})
            self.assertTrue(all(r in values for r in restored))
            self.assertTrue(all(len(text) < 400 for text in scored))
        finally:
            kb.close()

    def test_prose_completion_stops_at_paragraph_and_hard_limit(self):
        kb = self.open()
        try:
            for size in (1050, 1150):
                p = self.base / 'prose.txt'
                p.write_text('a' * size + '\n\nNEXT PARAGRAPH', encoding='utf-8')
                _, chunks = kb.chunk(read_document(p), 's', p.name)
                bodies = {c.metadata['reading_body'] for c in chunks}
                self.assertIn('NEXT PARAGRAPH', bodies)
                self.assertTrue(all(len(body) <= 1100 for body in bodies))
                self.assertTrue(all('NEXT PARAGRAPH' not in b or b == 'NEXT PARAGRAPH' for b in bodies))
                if size == 1050:
                    self.assertIn('a' * size, bodies)
                else:
                    self.assertNotIn('a' * size, bodies)
        finally:
            kb.close()
