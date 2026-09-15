"""Small real-model smoke check, not an AppWorld evaluation or paid API run."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_rag.service import KnowledgeBase, PROJECT


def main():
    root=PROJECT/'.agent/rag-checks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    root.mkdir(parents=True)
    directory=root/'inbox';directory.mkdir()
    (directory/'manual.md').write_text(
        '# API reference\n\n## SUM\nSUM adds numeric values. Use it to calculate the total of numbers.'
        '\n\n## Login\nLogin requires an access token. Expired credentials must be refreshed.'
        '\n\n## Playlists\nPlaylists contain songs. Query all pages to get the complete list.',encoding='utf-8')
    config=json.loads((PROJECT/'config/rag.json').read_text(encoding='utf-8'))
    config['root']=str(root/'index')
    kb=KnowledgeBase('smoke',config)
    try:
        result={'model':config['model'],'revision':config['revision'],'dimensions':config['dimensions'],
                'index':kb.sync(directory),'repeat':kb.sync(directory),'queries':[]}
        for query,expected in [('如何把数字求和？','SUM'),('凭据过期后怎么办？','Login'),('怎样获取歌单的全部歌曲？','Playlists')]:
            hits=kb.search(query)
            result['queries'].append({'query':query,'expected':expected,
                'actual':hits[0]['heading'] if hits else None,
                'passed':bool(hits and hits[0]['heading'].endswith('/ '+expected))})
        result['passed']=all(r['status']=='indexed' for r in result['index']) and all(r['status']=='unchanged' for r in result['repeat']) and all(q['passed'] for q in result['queries'])
        (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'evidence':str(root),'passed':result['passed'],'queries':result['queries']},ensure_ascii=False))
        if not result['passed']:
            raise RuntimeError('RAG smoke check failed; inspect evidence')
    finally:
        kb.close()

if __name__=='__main__':main()
