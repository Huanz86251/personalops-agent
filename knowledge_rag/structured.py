"""Format-aware parsing and bounded chunks, without generative model calls.

The original tree is retained separately from the compact retrieval text.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveJsonSplitter
from observability import trace_span, set_span_output


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def key(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


@lru_cache(maxsize=1)
def converter():
    from docling.document_converter import DocumentConverter
    return DocumentConverter()


def parse(path):
    path = Path(path)
    suffix = path.suffix.lower()
    with trace_span('RAG / Parse Structure', input_value={'source': path.name, 'suffix': suffix}) as span:
        if suffix in {'.json', '.jsonl', '.txt'}:
            text = path.read_text(encoding='utf-8-sig')
            if suffix == '.jsonl':
                value = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                try:
                    value = json.loads(text)
                except (ValueError, RecursionError):
                    if suffix == '.json':
                        raise ValueError('Invalid JSON; original index retained')
                    value = None
            if suffix != '.txt' or isinstance(value, (dict, list)):
                # Strict serialization also rejects NaN/Infinity.
                result = Document(page_content=compact(value), metadata={'parser': 'json'})
                set_span_output(span, {'parser': 'json', 'root_type': type(value).__name__})
                return [result]
            if not text.strip():
                return []
            # TXT remains prose. Embedded examples are not detached or guessed.
            from docling_core.types.doc import DoclingDocument, DocItemLabel
            doc = DoclingDocument(name=path.name)
            for paragraph in text.split('\n\n'):
                if paragraph.strip():
                    doc.add_text(label=DocItemLabel.PARAGRAPH, text=paragraph)
        else:
            result = converter().convert(path)
            if str(result.status.value) != 'success':
                raise ValueError(f'Docling conversion incomplete: {result.status}')
            doc = result.document
        content = doc.export_to_markdown()
        set_span_output(span, {'parser': 'docling', 'text_characters': len(content),
                              'items': len(doc.texts), 'tables': len(doc.tables)})
        return [Document(page_content=content, metadata={
            'parser': 'docling', 'docling_document': doc.model_dump(mode='json')})]


def display_path(parts, kb):
    """Keep the full path in metadata; abbreviate only the display prefix."""
    parts = [str(p) for p in parts if str(p)]
    full = ' / '.join(parts)
    limit = kb.config.get('path_tokens', 64)
    if len(parts) <= 6 and kb.count(full) <= limit:
        return full
    visible = [*parts[:3], '…', *parts[-3:]] if len(parts) > 6 else parts[:]
    if kb.count(' / '.join(visible)) <= limit:
        return ' / '.join(visible)
    # Individual labels can themselves exceed the entire budget.
    each = max(1, (limit - kb.count(' / '.join('' for _ in visible))) // len(visible))
    text = ' / '.join(kb._fit_text(p, each) for p in visible)
    return kb._fit_text(text, limit)


_API_METHODS = {'get', 'put', 'post', 'delete', 'patch', 'options', 'head', 'trace'}
_API_ACTIONS = {
    'show', 'get', 'list', 'search', 'find', 'read', 'fetch', 'create', 'add',
    'update', 'set', 'delete', 'remove', 'like', 'unlike', 'review', 'download',
    'upload', 'send', 'verify', 'accept', 'decline', 'cancel', 'login', 'logout',
    'signup', 'subscribe', 'unsubscribe', 'mark', 'move', 'copy', 'rename',
    'reset', 'follow', 'unfollow', 'play', 'pause', 'previous', 'next', 'seek',
    'loop', 'shuffle', 'clear',
}
_API_QUALIFIERS = {
    'liked', 'saved', 'favorite', 'favorites', 'private', 'privates', 'public',
    'library', 'libraries', 'detail', 'details', 'info', 'information', 'reviews',
    'review', 'history', 'histories',
    'current', 'downloaded', 'following', 'verification', 'to',
}


def _words(value):
    return [part.casefold() for part in re.findall(r'[A-Za-z][A-Za-z0-9]*', str(value or ''))]


def _singular(value):
    if value.endswith('ies') and len(value) > 4:
        return value[:-3] + 'y'
    if value.endswith('ses') and len(value) > 4:
        return value[:-2]
    if value.endswith('s') and not value.endswith('ss') and len(value) > 3:
        return value[:-1]
    return value


def _api_resource(record):
    """Derive a stable catalog resource without semantic generation."""
    tags = record.get('_catalog_tags') or record.get('tags') or []
    if isinstance(tags, list) and tags:
        return re.sub(r'[^a-z0-9_-]+', '-', str(tags[0]).casefold()).strip('-') or 'other'
    name_words = _words(record.get('api_name') or record.get('operationId'))
    candidates = [word for word in name_words
                  if word not in _API_ACTIONS and word not in _API_QUALIFIERS]
    if candidates:
        return _singular(candidates[0])
    path_words = [word for word in _words(record.get('path'))
                  if word not in _API_ACTIONS and word not in _API_QUALIFIERS]
    return _singular(path_words[-1]) if path_words else 'other'


def _api_operation(record):
    return 'read' if str(record.get('method', '')).casefold() in {'get', 'head', 'options'} else 'write'


def _api_records(value, filename):
    """Return normalized endpoint records for AppWorld JSON or OpenAPI."""
    if isinstance(value, list) and value and all(
            isinstance(item, dict) and item.get('api_name') and item.get('app_name')
            for item in value):
        return value
    # AppWorld's complete reference is {app_name: {api_name: endpoint_doc}}.
    # Accept only dictionaries whose leaves identify themselves as API records;
    # ordinary nested JSON must continue through the generic JSON parser.
    if isinstance(value, dict) and value:
        nested = []
        valid = True
        for app_name, endpoints in value.items():
            if not isinstance(endpoints, dict) or not endpoints:
                valid = False
                break
            for api_name, record in endpoints.items():
                if (not isinstance(record, dict) or not record.get('api_name')
                        or not record.get('app_name')):
                    valid = False
                    break
                normalized = dict(record)
                if (str(normalized['app_name']) != str(app_name)
                        or str(normalized['api_name']) != str(api_name)):
                    valid = False
                    break
                nested.append(normalized)
            if not valid:
                break
        if valid and nested:
            return nested
    if not isinstance(value, dict) or not isinstance(value.get('paths'), dict):
        return []
    title = str((value.get('info') or {}).get('title') or Path(filename).stem)
    records = []
    for path, path_item in value['paths'].items():
        if not isinstance(path_item, dict):
            continue
        shared_parameters = path_item.get('parameters') or []
        for method, operation in path_item.items():
            if method.casefold() not in _API_METHODS or not isinstance(operation, dict):
                continue
            operation_id = operation.get('operationId') or (
                method.casefold() + '_' + '_'.join(_words(path))
            )
            records.append({
                'app_name': title,
                'api_name': operation_id,
                'path': path,
                'method': method.upper(),
                'description': operation.get('summary') or operation.get('description') or '',
                'parameters': [*shared_parameters, *(operation.get('parameters') or [])],
                'requestBody': operation.get('requestBody'),
                'responses': operation.get('responses') or {},
                '_catalog_tags': operation.get('tags') or [],
            })
    return records


def chunk_documents(kb, documents, source_id, filename):
    parents, children = {}, []
    reading_mode = 'reading_block_tokens' in kb.config
    reading_limit = kb.config.get('reading_block_tokens', kb.config['child_tokens'])
    completion = kb.config.get('structure_completion_tokens', 0)

    def node(path, parent='', **extra):
        ident = key(source_id + compact(path))
        parents.setdefault(ident, {'parent_id': parent, 'path': path, 'source': filename, **extra})
        return ident

    root = node([filename])
    # Canonical node content is deduplicated by digest, outside vector payloads.
    def content(ident, text):
        parents[ident]['content'] = text

    def retain_tree(value, path, parent, depth=0):
        if depth > 80:
            raise ValueError('JSON nesting exceeds supported depth (80)')
        items = value.items() if isinstance(value,dict) else enumerate(value) if isinstance(value,list) else []
        for label,item in items:
            child_path = path + [str(label) if isinstance(value,dict) else f'[{label}]']
            child = node(child_path,parent,kind='json',value_type=type(item).__name__)
            content(child,compact(item))
            retain_tree(item,child_path,child,depth+1)

    def emit(body, path, parent, group, **metadata):
        prefix = display_path(path, kb)
        budget = min(kb.config['child_tokens'], kb.config.get('retrieval_chunk_tokens', 1200)-kb.count(prefix+'\n')-4)
        if budget < 1:
            raise ValueError('Path leaves no chunk body budget')
        if reading_mode:
            # Only a complete structural unit may use the completion allowance.
            blocks = [body] if kb.count(body) <= reading_limit + completion else kb._splitter(reading_limit).split_text(body)
            for block in blocks:
                block_id = key(source_id + compact(path) + str(len(children)) + block)
                windows = [block] if kb.count(block) <= budget else kb._splitter(budget, min(kb.config.get('overlap_tokens', 0), budget-1)).split_text(block)
                for part in windows:
                    text = prefix + '\n' + part
                    if kb.count(text) > kb.config['embedding_max_tokens'] - 32:
                        raise ValueError('Retrieval window exceeds embedding capacity')
                    children.append(Document(page_content=text, metadata={
                        'source_id': source_id, 'source': filename, 'parent_id': parent,
                        'record_id': block_id, 'reading_block_id': block_id,
                        'reading_body': block, 'reading_body_tokens': kb.count(block),
                        'completion_tokens': max(0, kb.count(block)-reading_limit),
                        'heading': ' / '.join(path[1:]), 'full_path': path,
                        'prefix': prefix, 'path_abbreviated': prefix != ' / '.join(path),
                        'body': part, 'oversized_split': len(blocks)>1, **metadata}))
            return
        parts = [body] if kb.count(body) <= budget else kb._splitter(budget).split_text(body)
        for part in parts:
            text = prefix + '\n' + part
            if kb.count(text) > min(kb.config['embedding_max_tokens']-32, kb.config.get('retrieval_chunk_tokens',1200)):
                raise ValueError('Structured chunk exceeds token budget')
            children.append(Document(page_content=text, metadata={
                'source_id': source_id, 'source': filename, 'parent_id': parent,
                'record_id': group, 'heading': ' / '.join(path[1:]), 'full_path': path,
                'prefix': prefix, 'path_abbreviated': prefix != ' / '.join(path),
                'body': part, 'oversized_split': len(parts)>1, **metadata}))

    def api_catalog(records):
        """Index compact endpoint cards and retain full endpoints as readable leaves."""
        app_nodes, group_nodes, group_entries = {}, {}, {}
        for record in records:
            app = str(record.get('app_name') or Path(filename).stem)
            api_name = str(record['api_name'])
            resource = _api_resource(record)
            operation = _api_operation(record)
            app_id = app_nodes.setdefault(app, node([filename, app], root, kind='api_app'))
            resource_id = node([filename, app, resource], app_id, kind='api_resource')
            group_path = [filename, app, resource, operation]
            group_id = group_nodes.setdefault(
                (app, resource, operation),
                node(group_path, resource_id, kind='api_catalog'),
            )
            endpoint_path = [*group_path, api_name]
            endpoint_id = node(
                endpoint_path, group_id, kind='api_endpoint',
                identity=[f'api_name={api_name}'], api_name=api_name,
                method=str(record.get('method') or ''),
                api_path=str(record.get('path') or ''),
                description=str(record.get('description') or ''),
            )
            stored = {k: v for k, v in record.items() if not k.startswith('_')}
            content(endpoint_id, compact(stored))
            entry = {
                'node_id': endpoint_id,
                'relative_path': '/'.join(endpoint_path[1:]),
                'api_name': api_name,
                'method': str(record.get('method') or ''),
                'path': str(record.get('path') or ''),
                'description': str(record.get('description') or ''),
            }
            group_entries.setdefault(group_id, []).append(entry)
            card = compact({k: entry[k] for k in ('api_name', 'method', 'path', 'description')})
            prefix = display_path(endpoint_path, kb)
            text = prefix + '\n' + card
            if kb.count(text) > kb.config['embedding_max_tokens'] - 32:
                text = kb._fit_text(text, kb.config['embedding_max_tokens'] - 32)
            children.append(Document(page_content=text, metadata={
                'source_id': source_id, 'source': filename, 'parent_id': endpoint_id,
                'catalog_parent_id': group_id, 'leaf_node_id': endpoint_id,
                'record_id': endpoint_id, 'retrieval_kind': 'api_endpoint',
                'api_name': api_name, 'api_resource': resource,
                'api_operation': operation,
                'heading': ' / '.join(endpoint_path[1:]), 'full_path': endpoint_path,
                'prefix': prefix, 'path_abbreviated': prefix != ' / '.join(endpoint_path),
                'body': card, 'parser': 'api_json', 'oversized_split': False,
            }))
        for group_id, entries in group_entries.items():
            entries.sort(key=lambda item: item['api_name'])
            parents[group_id]['entries'] = entries
            content(group_id, '\n'.join(
                f"{item['api_name']} — {item['description']}" for item in entries
            ))

    def json_walk(value, path, parent, group='', depth=0, in_record=False):
        if depth > 80:
            raise ValueError('JSON nesting exceeds supported depth (80)')
        current = node(path, parent, kind='json')
        content(current,compact(value))
        group = group or current
        if isinstance(value, dict):
            identities = [f'{k}={v}' for k,v in value.items()
                          if isinstance(v,(str,int)) and (k in {'name','title','id','operationId'} or k.endswith('_name'))]
            if identities:
                path = path + ['; '.join(identities)]
                parents[current]['identity'] = identities
        serialized = compact(value)
        # Never pack separate object records merely because their shared array is short.
        records = isinstance(value,list) and any(isinstance(v,(dict,list)) for v in value)
        if kb.count(serialized) <= reading_limit + completion and not records:
            retain_tree(value,path,current,depth)
            emit(serialized, path, current, group, parser='json', json_path=parents[current]['path'])
        elif isinstance(value, list):
            for i,item in enumerate(value):
                # Top-level record identity also drives retrieval deduplication.
                record_group = key(source_id+compact(path+[f'[{i}]'])) if not in_record else group
                json_walk(item,path+[f'[{i}]'],current,record_group,depth+1,in_record or isinstance(item,dict))
        elif isinstance(value, dict):
            # Existing splitter packs small sibling fields, preserving JSON structure.
            scalars = {k:v for k,v in value.items() if not isinstance(v,(dict,list))}
            retain_tree(scalars,path,current,depth)
            for label in list(scalars):
                item = scalars[label]
                if isinstance(item,str) and kb.count(compact({label:item})) > kb.config['child_tokens']:
                    emit(item,path+[label],current,group,parser='json',json_path=path+[label],fragment_format='text_value')
                    del scalars[label]
            if scalars:
                splitter = RecursiveJsonSplitter(max_chunk_size=max(32,kb.config['child_tokens']*3))
                for part in splitter.split_json(scalars):
                    if part:
                        emit(compact(part),path,current,group,parser='json',json_path=parents[current]['path'])
            for label,item in value.items():
                if isinstance(item,(dict,list)):
                    child_group = '' if group == root else group
                    json_walk(item,path+[label],current,child_group,depth+1,in_record)
        else:
            emit(serialized,path,current,group,parser='json',json_path=parents[current]['path'])

    for ordinal, document in enumerate(documents):
        parser = document.metadata.get('parser')
        if parser == 'json':
            value = json.loads(document.page_content)
            records = _api_records(value, filename)
            if records:
                api_catalog(records)
            else:
                json_walk(value,[filename], '', depth=0)
            continue
        from docling_core.types.doc import DoclingDocument, DocItemLabel
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
        from pydantic import ConfigDict
        class CountingTokenizer(BaseTokenizer):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            def count_tokens(self, text): return kb.count(text)
            def get_max_tokens(self): return kb.config['child_tokens']
            def get_tokenizer(self): return kb.count
        if 'docling_document' in document.metadata:
            doc = DoclingDocument.model_validate(document.metadata['docling_document'])
        else:
            # Direct callers supplying Markdown still use Docling's parser.
            from docling.datamodel.base_models import DocumentStream
            doc = converter().convert(DocumentStream(name='input.md',stream=io.BytesIO(document.page_content.encode()))).document
        if reading_mode:
            from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
            chunker = HierarchicalChunker(merge_list_items=False)
        else:
            chunker = HybridChunker(tokenizer=CountingTokenizer(), merge_peers=True)
        content(root,doc.export_to_markdown())
        section_for_item = {}
        section_ref = 'body'
        for item, _ in doc.iterate_items():
            label = getattr(item, 'label', '')
            if getattr(label, 'value', label) in {'section_header', 'title'}:
                section_ref = item.self_ref
            section_for_item[item.self_ref] = section_ref
        for chunk in chunker.chunk(dl_doc=doc):
            headings = list(chunk.meta.headings or [])
            path = [filename, *headings]
            parent = root
            for i in range(1,len(path)):
                parent = node(path[:i+1],parent,kind='section')
            if parent == root:
                parent = node([filename,f'document-{ordinal}'],root,kind='section')
            items = chunk.meta.doc_items or []
            if items:
                ref = section_for_item.get(items[0].self_ref,items[0].self_ref)
                parent = node(path+['@'+ref],parent,kind='section_occurrence',element_ref=ref)
            # Collect each section once in original order, before retrieval splitting.
            visited=set()
            ancestor=parent
            while ancestor and ancestor != root and ancestor not in visited:
                visited.add(ancestor)
                section_text = (' / '.join(headings)+'\n'+chunk.text).strip()
                parents[ancestor]['content'] = (parents[ancestor].get('content','')+'\n\n'+section_text).strip()
                ancestor=parents[ancestor]['parent_id']
            pages = sorted({p.page_no for item in items for p in item.prov})
            emit(chunk.text,path,parent,parent,parser='docling',pages=pages,
                 element_refs=[item.self_ref for item in items])
    for i,child in enumerate(children):
        child.metadata['chunk_id'] = key(source_id+str(i)+child.page_content)
    for i,child in enumerate(children):
        for label,j in [('previous_chunk_id',i-1),('next_chunk_id',i+1)]:
            child.metadata[label] = children[j].metadata['chunk_id'] if 0 <= j < len(children) and children[j].metadata['parent_id']==child.metadata['parent_id'] else ''
    return parents,children
