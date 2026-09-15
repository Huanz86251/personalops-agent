from knowledge_rag.guidance import format_rag_context, read_hint, search_hint


def test_next_page_keeps_original_parent_anchor():
    result = read_hint({'status':'ok','next_offset':913,'node_id':'resolved-parent'},
        reference_id='real-ref',offset=0,view='content',node_id='original-node',depth=2)
    args=result['example_call']['arguments']
    assert args['node_id']=='original-node' and args['offset']==913


def test_outline_uses_returned_node_and_resets_offset():
    result=read_hint({'status':'ok','next_offset':None,'entries':[{'node_id':'actual-child','readable':True}]},
        reference_id='real-ref',offset=40,view='children',node_id=None,depth=4)
    args=result['example_call']['arguments']
    assert args['view']=='content' and args['node_id']=='actual-child' and args['offset']==0


def test_search_real_reference_and_empty_result():
    hit=search_hint({'status':'ok','rag':[{'reference_id':'real-ref','node_id':'node'}]})
    assert hit['example_call']['arguments']['reference_id']=='real-ref'
    empty=search_hint({'status':'ok','rag':[],'memory':''})
    assert 'example_call' not in empty and '未命中' in empty['usage_hint']
    memory=search_hint({'status':'ok','memory':'preference'})
    assert 'example_call' not in memory


def test_api_catalog_example_uses_short_exact_api_name():
    hit=search_hint({'status':'ok','rag':[{
        'kind':'api_catalog','reference_id':'real-ref','node_id':'parent',
        'entries':[{'api_name':'show_orders','node_id':'long-leaf-id'}],
    }]})
    assert hit['example_call']['arguments'] == {
        'reference_id':'real-ref','api_name':'show_orders','view':'content'}


def test_errors_do_not_offer_invalid_read_example():
    result=read_hint({'status':'stale_reference'},reference_id='old',
                     offset=0,view='content',node_id=None,depth=2)
    assert 'example_call' not in result


def test_completed_leaf_browses_its_parent_instead_of_empty_leaf_children():
    result=read_hint({'status':'ok','next_offset':None,'node_id':'leaf','parent_id':'catalog'},
                     reference_id='real-ref',offset=0,view='content',node_id='leaf',depth=2)
    args=result['example_call']['arguments']
    assert args == {'reference_id':'real-ref','node_id':'catalog','view':'children','depth':1,'offset':0}


def test_completed_root_without_parent_does_not_offer_useless_children_call():
    result=read_hint({'status':'ok','next_offset':None,'node_id':'root','parent_id':None},
                     reference_id='real-ref',offset=0,view='content',node_id='root',depth=2)
    assert 'example_call' not in result
    assert '无需重复' in result['usage_hint']


def test_rag_scope_guidance_follows_results_and_names_subsets():
    text = format_rag_context([{'source': 'guide.md', 'text': 'content'}])
    assert text.index('guide.md') < text.index('[RAG使用提醒]')
    assert '可能过时、不完整' in text
    assert '我关注的项目' in text and '不能代替所有项目' in text
    assert 'reference_id' in text and 'root_node_id' in text
    assert 'content' in text and 'children' in text
    assert format_rag_context([]) == ''
