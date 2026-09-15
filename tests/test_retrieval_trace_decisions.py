import unittest
from unittest.mock import AsyncMock,patch
from types import SimpleNamespace as NS
from contextlib import contextmanager
from memory import MemoryService,RetrievedMemory


class DecisionTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_confidence_reorders_and_trace_explains_rejection(self):
        service=object.__new__(MemoryService)
        service.final_limit=1;service.reranker_threshold=.4
        service.retrieval_models=NS(arerank=AsyncMock(return_value=[NS(index=0,score=.9),NS(index=1,score=.85),NS(index=2,score=.3)]))
        values=[RetrievedMemory(memory_id=str(i),content='fixture '+str(i),memory_type='semantic',importance=2,confidence=c) for i,c in enumerate([1,3,4])]
        outputs={}
        @contextmanager
        def span(name,**kwargs):yield name
        with patch('memory.trace_span',span),patch('memory.set_span_output',side_effect=lambda span,value:outputs.update({span:value})):
            selected=await service._rerank('fixture',values)
            rendered=service.format_context(selected)
        self.assertEqual([x.memory_id for x in selected],['1'])
        rows=outputs['Memory / Ranking Decisions']['candidates']
        self.assertAlmostEqual(rows[0]['weighted_score'],.72)
        self.assertEqual(rows[0]['decision'],'outside_top_k')
        self.assertEqual(rows[1]['decision'],'selected')
        self.assertEqual(rows[2]['decision'],'below_threshold')
        self.assertEqual(outputs['Memory / Assemble Context']['context'],rendered)
        self.assertEqual(outputs['Memory / Assemble Context']['memory_ids'],['1'])
