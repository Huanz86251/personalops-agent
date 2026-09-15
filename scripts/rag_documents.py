"""Local RAG entry: scan new/changed files before each query. No chat API."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_rag.service import KnowledgeBase, PROJECT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="local")
    parser.add_argument("--directory")
    parser.add_argument("--query")
    args = parser.parse_args()
    kb = KnowledgeBase(scope=args.scope)
    try:
        directory = Path(args.directory) if args.directory else PROJECT / kb.config["inbox"]
        directory.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"ingestion": kb.sync(directory)}, ensure_ascii=False, indent=2))
        if args.query:
            print(json.dumps({"results": kb.search(args.query)}, ensure_ascii=False, indent=2))
    finally:
        kb.close()

if __name__ == "__main__":
    main()
