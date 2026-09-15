"""Scoped, versioned read grants: no path supplied by the model is opened."""
import json
import sqlite3
from contextlib import closing
from uuid import uuid4


def public_result(result):
    # Complete paths remain in storage/trace, not duplicated in model context.
    hidden = {'full_path','heading'}
    # API catalogs already expose structured entries. Repeating the same list as
    # a rendered text block wastes context and makes the interface look ambiguous.
    if result.get('kind') == 'api_catalog':
        hidden.update({
            'text', 'node_id', 'parent_id', 'root_node_id',
            'relevance_score', 'context_tokens', 'catalog_only',
        })
    visible = {k:v for k,v in result.items() if not k.startswith('_') and k not in hidden}
    if result.get('kind') == 'api_catalog':
        # The model reads an endpoint by exact api_name. Long opaque leaf IDs
        # remain private in the grant/manifest and need not pollute discovery.
        visible['entries'] = [
            {k: v for k, v in entry.items() if k != 'node_id'}
            for entry in visible.get('entries', [])
        ]
    return visible


def grants(hub, run, agent, results=None):
    with closing(sqlite3.connect(hub.root/'knowledge-grants.sqlite3')) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS grants (run TEXT, agent TEXT, ref TEXT, payload TEXT, PRIMARY KEY(run,agent,ref))')
        if results is not None:
            visible=[]
            for result in results:
                if (result.get('relevance_score') or 0) < hub.config.get('relevance_threshold',.4): continue
                if not all(k in result for k in ('_source_id','_version','_collection','parent_id')): continue
                ref=uuid4().hex
                db.execute('INSERT INTO grants VALUES (?,?,?,?)',(run,agent,ref,json.dumps(result,ensure_ascii=False)))
                visible.append(public_result(result)|{'reference_id':ref})
            return visible
        return {ref:json.loads(payload) for ref,payload in db.execute('SELECT ref,payload FROM grants WHERE run=? AND agent=? ORDER BY rowid',(run,agent))}


def read_page(kb, grant, offset, *, view='content', node_id=None, api_name=None, depth=2):
    if offset < 0: raise ValueError('offset must be nonnegative')
    if view not in {'content','children'} or not 1<=depth<=8:
        raise ValueError('view must be content/children; depth must be 1–8')
    entry=kb.manifest.get(grant['_source_id'])
    if not entry or entry['hash'] != grant['_version'] or kb.collection != grant['_collection']:
        return {'status':'stale_reference','message':'文档版本已变化，请重新检索。'}
    nodes=entry['parents']
    if node_id is not None and api_name is not None:
        raise ValueError('node_id and api_name cannot be used together')
    if api_name is not None:
        if view != 'content':
            raise ValueError('api_name only supports content view')
        matches = [
            ident for ident, node in nodes.items()
            if node.get('parent_id') == grant['parent_id']
            and node.get('kind') == 'api_endpoint'
            and node.get('api_name') == api_name
        ]
        if len(matches) != 1:
            return {'status':'invalid_api_name','message':'该父目录下没有这个精确api_name，请从api_names中原样复制。'}
        node_id = matches[0]
    if node_id and node_id not in nodes:return {'status':'invalid_node'}
    ident=node_id or grant['parent_id']
    if view=='children':
        return outline_page(kb,nodes,ident,depth,offset,grant.get('full_path',[]))
    node=nodes[ident]
    if 'content_ref' not in node: return {'status':'unavailable','message':'该节点没有原文，请重新索引。'}
    text=(kb.root/'node-content'/(node['content_ref']+'.txt')).read_text(encoding='utf-8')
    if offset > len(text): raise ValueError('Page offset is outside document')
    budget=min(2000,kb.config.get('read_page_tokens',2000))
    page=kb._fit_text(text[offset:],budget)
    if offset+len(page)<len(text):
        boundary=page.rfind('\n')
        if boundary>=len(page)//2:page=page[:boundary+1]
    if offset < len(text) and not page: raise ValueError('Page budget too small')
    end=offset+len(page)
    from knowledge_rag.structured import display_path
    return {'status':'ok','view':'content','node_id':ident,'parent_id':node.get('parent_id') or None,
            'text':page,'tokens':kb.count(page),'path':display_path(node['path'],kb),
            'offset':offset,'next_offset':end if end<len(text) else None,
            'complete':end==len(text),'source':grant['source'],'version':grant['_version']}


def outline_page(kb,nodes,ident,depth,offset,matched_path=()):
    """Directory listing only: no node text blobs loaded, bounded output."""
    children={}
    for child,node in nodes.items():
        if node.get('kind')=='section_occurrence':continue
        children.setdefault(node.get('parent_id',''),[]).append(child)
    from knowledge_rag.structured import display_path
    path=' / '.join(nodes[ident]['path'])
    bounded_path=kb._fit_text(path,400)
    result={'status':'ok','view':'children','node_id':ident,'path':bounded_path,
            'path_truncated':bounded_path!=path,'depth':depth,'offset':offset,'entries':[]}
    matched=' / '.join(matched_path)
    result['matched_path']=kb._fit_text(matched,400)
    result['matched_path_truncated']=result['matched_path']!=matched
    stack=[(c,1) for c in reversed(children.get(ident,[]))]
    visited=set();position=0;more=False
    while stack:
        child,level=stack.pop()
        if child in visited:raise ValueError('Cyclic document structure')
        visited.add(child)
        if position<offset:
            position+=1
            if level<depth:stack.extend((c,level+1) for c in reversed(children.get(child,[])))
            continue
        node=nodes[child]
        label=str(node['path'][-1])
        identity='; '.join(node.get('identity',[]))
        if identity and identity not in label: label += ' — ' + identity
        short=kb._fit_text(label,100)
        item={'node_id':child,'title':short,'title_truncated':short!=label,'depth':level,
              'has_children':bool(children.get(child)),'readable':'content_ref' in node}
        tentative={**result,'entries':result['entries']+[item]}
        if len(result['entries'])>=40 or kb.count(json.dumps(tentative,ensure_ascii=False))>1800:
            more=True;break
        result['entries'].append(item);position+=1
        if level<depth:stack.extend((c,level+1) for c in reversed(children.get(child,[])))
    if more and not result['entries']:raise ValueError('Directory entry exceeds page budget')
    if offset>position:raise ValueError('Outline offset is outside listing')
    result.update(next_offset=position if more else None,complete=not more)
    result['tokens']=kb.count(json.dumps(result,ensure_ascii=False))
    return result
