"""Real parsers with deterministic token counting; no paid model requests."""
import json
from langchain_core.documents import Document
from knowledge_rag.service import read_document
from test_knowledge_rag import RagTests


class StructuredTests(RagTests):
    def parsed(self, name, text):
        path=self.base/name
        path.write_text(text,encoding='utf-8')
        kb=self.open()
        try:
            return kb.chunk(read_document(path),'fixture',name)
        finally: kb.close()

    def test_json_records_never_mix_and_keep_identity(self):
        records=[{'name':name,'description':'purpose '*40,'parameters':{'rating':{'type':'integer','min':1,'max':5}}} for name in ('create','update')]
        parents,chunks=self.parsed('api.json',json.dumps(records))
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertNotEqual('name=create' in chunk.page_content,'name=update' in chunk.page_content)
            self.assertIn(chunk.metadata['parent_id'],parents)
            self.assertLessEqual(len(chunk.page_content),400)
        self.assertEqual(len({c.metadata['record_id'] for c in chunks}),2)

    def test_nested_small_object_preserves_tree_but_one_chunk(self):
        parents,chunks=self.parsed('config.json','{"a":{"b":{"c":1}}}')
        self.assertEqual(len(chunks),1)
        self.assertTrue(any(p['path'][-1]=='c' for p in parents.values()))
        self.assertTrue(all(not p['parent_id'] or p['parent_id'] in parents for p in parents.values()))

    def test_txt_json_detected_mixed_example_kept_as_prose(self):
        _,chunks=self.parsed('config.txt','{"a":1}')
        self.assertEqual(chunks[0].metadata['parser'],'json')
        _,chunks=self.parsed('mixed.txt','Example follows:\n{"a":1}\nThis means success.')
        self.assertTrue(all(c.metadata['parser']=='docling' for c in chunks))
        self.assertIn('This means success',' '.join(c.page_content for c in chunks))

    def test_invalid_json_rejected(self):
        with self.assertRaises(ValueError): self.parsed('bad.json','{"a":')

    def test_long_string_retains_field_identity(self):
        _,chunks=self.parsed('config.json',json.dumps({'description':'sentence '*100}))
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(c.metadata['full_path'][-1]=='description' for c in chunks))
        self.assertTrue(all(len(c.page_content)<=400 for c in chunks))

    def test_markdown_heading_path_and_html(self):
        _,chunks=self.parsed('guide.md','# Manual\n\n## Commands\n\n### SUM\n\nAdds numbers.')
        self.assertEqual(chunks[0].metadata['full_path'],['guide.md','Manual','Commands','SUM'])
        _,chunks=self.parsed('guide.html','<html><body><h1>Manual</h1><h2>SUM</h2><p>Adds numbers.</p></body></html>')
        self.assertIn('Adds numbers',' '.join(c.page_content for c in chunks))

    def test_jsonl_record_boundary(self):
        _,chunks=self.parsed('events.jsonl','{"name":"first"}\n{"name":"second"}')
        self.assertEqual(len(chunks),2)
        self.assertNotEqual(chunks[0].metadata['record_id'],chunks[1].metadata['record_id'])

    def test_large_wrapped_json_keeps_all_records(self):
        data={'records':[{'id':i,'description':'payload '+str(i)} for i in range(250)]}
        parents,chunks=self.parsed('large.json',json.dumps(data))
        self.assertEqual(len({c.metadata['record_id'] for c in chunks}),250)
        self.assertTrue(all(not p['parent_id'] or p['parent_id'] in parents for p in parents.values()))

    def test_deep_path_abbreviated_not_deleted(self):
        data={'leaf':'answer'}
        for i in range(20):data={f'level_{i}':data}
        _,chunks=self.parsed('deep.json',json.dumps(data))
        self.assertTrue(any(c.metadata['path_abbreviated'] for c in chunks))
        self.assertTrue(any(len(c.metadata['full_path'])>10 for c in chunks))
        self.assertTrue(all(len(c.page_content)<=400 for c in chunks))
