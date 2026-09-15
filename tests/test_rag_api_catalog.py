import json
from types import SimpleNamespace as NS

from langchain_core.documents import Document

from knowledge_rag.expansion import read_page, public_result
from test_knowledge_rag import RagTests


class ApiCatalogTests(RagTests):
    def _write_reference(self):
        path = self.base / 'api-reference.json'
        path.write_text(json.dumps({
            'spotify': {
                'show_liked_playlists': {
                    'app_name': 'spotify', 'api_name': 'show_liked_playlists',
                    'method': 'GET', 'path': '/liked-playlists',
                    'description': 'Show playlists liked by the current user.',
                    'parameters': [{'name': 'access_token', 'required': True}],
                },
                'show_playlist_library': {
                    'app_name': 'spotify', 'api_name': 'show_playlist_library',
                    'method': 'GET', 'path': '/playlist-library',
                    'description': 'Show playlists owned by the current user.',
                    'parameters': [{'name': 'access_token', 'required': True}],
                },
                'create_playlist': {
                    'app_name': 'spotify', 'api_name': 'create_playlist',
                    'method': 'POST', 'path': '/playlists',
                    'description': 'Create a playlist.',
                    'parameters': [{'name': 'title', 'required': True}],
                },
            }
        }), encoding='utf-8')
        return path

    def test_semantic_api_hit_returns_parent_catalog_and_paths(self):
        kb = self.open()
        try:
            kb.ingest(self._write_reference())
            hits = kb.search('find playlists in my library', scorer=lambda q, docs: [
                NS(index=i, score=.9 if 'playlist' in text.casefold() else .1)
                for i, text in enumerate(docs)
            ])
            catalog = next(hit for hit in hits if hit['kind'] == 'api_catalog'
                           and any(entry['api_name'] == 'show_playlist_library'
                                   for entry in hit['entries']))
            names = {entry['api_name'] for entry in catalog['entries']}
            self.assertIn('show_playlist_library', names)
            self.assertIn('show_liked_playlists', catalog['api_names'])
            self.assertNotIn('parameters', catalog['text'])
            self.assertTrue(all(entry['node_id'] and entry['relative_path'] for entry in catalog['entries']))
        finally:
            kb.close()

    def test_exact_api_name_is_the_only_leaf_shortcut(self):
        kb = self.open()
        try:
            kb.ingest(self._write_reference())
            hits = kb.search('Call show_playlist_library for the current account')
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]['kind'], 'api_endpoint')
            self.assertEqual(hits[0]['match'], 'exact_api_name')
            endpoint = json.loads(hits[0]['text'])
            self.assertEqual(endpoint['api_name'], 'show_playlist_library')
            self.assertEqual(endpoint['parameters'][0]['name'], 'access_token')
        finally:
            kb.close()

    def test_catalog_node_can_open_exact_endpoint(self):
        kb = self.open()
        try:
            kb.ingest(self._write_reference())
            hit = kb.search('playlist library', scorer=lambda q, docs: [
                NS(index=i, score=.9 if 'playlist' in text.casefold() else .1)
                for i, text in enumerate(docs)
            ])[0]
            target = next(entry for entry in hit['entries']
                          if entry['api_name'] == 'show_playlist_library')
            page = read_page(kb, hit, 0, node_id=target['node_id'], view='content')
            self.assertEqual(page['status'], 'ok')
            self.assertEqual(json.loads(page['text'])['api_name'], 'show_playlist_library')
            self.assertEqual(page['parent_id'], hit['node_id'])
        finally:
            kb.close()

    def test_public_catalog_hides_long_leaf_ids(self):
        visible = public_result({
            'kind': 'api_catalog',
            'text': 'duplicate rendering',
            'entries': [{
                'node_id': 'opaque-leaf-id',
                'api_name': 'show_playlist_library',
                'description': 'Show owned playlists.',
            }],
            'api_names': ['show_playlist_library'],
        })
        self.assertNotIn('text', visible)
        self.assertNotIn('node_id', visible)
        self.assertNotIn('root_node_id', visible)
        self.assertNotIn('relevance_score', visible)
        self.assertNotIn('context_tokens', visible)
        self.assertNotIn('node_id', visible['entries'][0])
        self.assertEqual(visible['entries'][0]['api_name'], 'show_playlist_library')

    def test_catalog_details_follow_global_rank_and_stop_at_four(self):
        path = self.base / 'ranked-api-reference.json'
        path.write_text(json.dumps({'spotify': {
            f'show_playlist_{index}': {
                'app_name': 'spotify',
                'api_name': f'show_playlist_{index}',
                'method': 'GET',
                'path': f'/playlists/{index}',
                'description': f'Playlist rank candidate {index}.',
            }
            for index in range(6)
        }}), encoding='utf-8')
        kb = self.open()
        try:
            kb.ingest(path)
            wanted = [4, 1, 5, 0, 3, 2]
            def scorer(query, docs):
                scores = {value: .95 - rank * .05 for rank, value in enumerate(wanted)}
                return [
                    NS(index=i, score=next(
                        score for value, score in scores.items()
                        if f'show_playlist_{value}' in text
                    ))
                    for i, text in enumerate(docs)
                ]
            catalog = kb.search('playlist candidates', scorer=scorer)[0]
            self.assertEqual(
                [entry['api_name'] for entry in catalog['entries']],
                [f'show_playlist_{value}' for value in wanted[:4]],
            )
            self.assertEqual(len(catalog['api_names']), 6)
            self.assertTrue(catalog['catalog_only'])
        finally:
            kb.close()

    def test_catalog_can_open_exact_endpoint_by_api_name(self):
        kb = self.open()
        try:
            kb.ingest(self._write_reference())
            hit = kb.search('playlist library', scorer=lambda q, docs: [
                NS(index=i, score=.9 if 'playlist' in text.casefold() else .1)
                for i, text in enumerate(docs)
            ])[0]
            page = read_page(kb, hit, 0, api_name='show_playlist_library')
            self.assertEqual(page['status'], 'ok')
            self.assertEqual(json.loads(page['text'])['api_name'], 'show_playlist_library')
            self.assertEqual(
                read_page(kb, hit, 0, api_name='guessed_api')['status'],
                'invalid_api_name',
            )
        finally:
            kb.close()

    def test_openapi_paths_build_the_same_catalog_without_a_model(self):
        kb = self.open()
        try:
            value = {
                'info': {'title': 'Store'},
                'paths': {'/orders/{id}': {'get': {
                    'operationId': 'get_order', 'tags': ['orders'],
                    'summary': 'Read one order',
                    'parameters': [{'name': 'id', 'in': 'path', 'required': True}],
                }}},
            }
            parents, chunks = kb.chunk([
                Document(page_content=json.dumps(value), metadata={'parser': 'json'})
            ], 'openapi', 'openapi.json')
            leaf = next(chunk for chunk in chunks if chunk.metadata.get('api_name') == 'get_order')
            catalog = parents[leaf.metadata['catalog_parent_id']]
            self.assertEqual(catalog['kind'], 'api_catalog')
            self.assertEqual(catalog['path'][-2:], ['orders', 'read'])
        finally:
            kb.close()
