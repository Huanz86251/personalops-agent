import unittest
from types import SimpleNamespace as NS
from reranker_inputs import effective_limit,describe_pairs


class Tokenizer:
    model_max_length=1024
    def num_special_tokens_to_add(self,pair):return 4
    def __call__(self,q,d,**kw):
        ids=list(range(len(q)+len(d)+4))
        if kw.get('truncation'):ids=ids[:kw['max_length']]
        return {'input_ids':ids}
    def decode(self,ids,**kw):return str(ids)


class RerankerLengthTests(unittest.TestCase):
    def model(self,kind,positions,pad=1):
        return NS(model=NS(config=NS(model_type=kind,max_position_embeddings=positions,pad_token_id=pad)),tokenizer=Tokenizer())
    def test_roberta_offsets_and_special_tokens(self):
        model=self.model('xlm-roberta',514)
        self.assertEqual(effective_limit(model,1024),512)
        details=describe_pairs(model,[('q'*40,'d'*486),('q','d')],512)
        self.assertEqual(details[0]['tokens_before'],530)
        self.assertEqual(details[0]['tokens_after'],512)
        self.assertTrue(details[0]['truncated'])
        self.assertFalse(details[1]['truncated'])
    def test_bert_has_no_roberta_position_offset(self):
        self.assertEqual(effective_limit(self.model('bert',512,0),1024),512)
        self.assertEqual(effective_limit(self.model('xlm-roberta',514),256),256)
    def test_tokenizer_limit_and_invalid_budget(self):
        model=self.model('bert',2048);model.tokenizer.model_max_length=128
        self.assertEqual(effective_limit(model,1024),128)
        with self.assertRaises(ValueError):effective_limit(model,2)


class CharTokenizer:
    def num_special_tokens_to_add(self,pair=True): return 4
    def encode(self,text,**kw): return list(map(ord,text))
    def decode(self,ids,**kw): return ''.join(map(chr,ids))
    def __call__(self,q,d,**kw): return {'input_ids':[0]*4+self.encode(q)+self.encode(d)}

class QueryFitTests(unittest.TestCase):
    def test_query_fit_preserves_both_ends_and_every_pair_fits(self):
        from reranker_inputs import fit_query_head_tail
        tokenizer = CharTokenizer()
        model = NS(tokenizer=tokenizer)
        original = "HEAD" + "x" * 30 + "TAIL"
        documents = ["d" * 8, "short"]
        fitted = fit_query_head_tail(model, original, documents, 24)
        self.assertTrue(fitted.startswith("HEAD"))
        self.assertTrue(fitted.endswith("TAIL"))
        self.assertTrue(all(len(tokenizer(fitted, doc)['input_ids']) <= 24 for doc in documents))


class WindowTests(unittest.TestCase):
    def test_two_windows_preserve_tail_and_bound_pairs(self):
        from reranker_inputs import document_windows
        model=NS(tokenizer=CharTokenizer())
        q,parts,owners,audit=document_windows(model,'query',['a'*12+'TAIL','short'],20,2)
        self.assertEqual(''.join(parts[:2]),'a'*12+'TAIL')
        self.assertEqual(owners,[0,0,1])
        self.assertTrue(all(len(q)+len(p)+4<=20 for p in parts))
        self.assertEqual(audit[1]['remaining_tokens'],0)
    def test_overflow_is_explicit_and_query_bounded(self):
        from reranker_inputs import document_windows
        q,parts,owners,audit=document_windows(NS(tokenizer=CharTokenizer()),'q'*50,['x'*100],20,2)
        self.assertEqual(len(parts),2)
        self.assertGreater(audit[-1]['remaining_tokens'],0)
        self.assertLess(len(q),50)

    def test_max_score_maps_back_to_original_document(self):
        from retrieval_models import RetrievalModelManager, RerankResult
        manager=RetrievalModelManager.__new__(RetrievalModelManager)
        manager.reranker_max_length=20
        manager._reranker_model=NS(tokenizer=CharTokenizer(), model=NS(config=NS(max_position_embeddings=20,model_type='bert')))
        manager.rerank=lambda q,docs,top_k: [RerankResult(index=i,text=d,score=.95 if 'TAIL' in d else .1) for i,d in enumerate(docs)]
        result=manager.rerank_windows('q',['a'*16+'TAIL','small'],top_k=2)
        self.assertEqual(result[0].index,0)
        self.assertEqual(result[0].score,.95)
        self.assertEqual(result[0].text,'a'*16+'TAIL')
