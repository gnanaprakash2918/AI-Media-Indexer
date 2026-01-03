#!/usr/bin/env python
import sys
import argparse
import asyncio
import json

from core.retrieval.engine import get_search_engine
from core.storage.db import VectorDB


async def run_search(query: str, use_rerank: bool, limit: int, as_json: bool) -> None:
    db = VectorDB(backend="docker")
    engine = get_search_engine(db=db)
    
    try:
        results = await engine.search(query=query, use_rerank=use_rerank, limit=limit)
        
        if as_json:
            output = [r.model_dump() for r in results]
            print(json.dumps(output, indent=2))
            return
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Results: {len(results)}")
        print(f"{'='*60}\n")
        
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r.video_path}")
            print(f"    Time: {r.start_time:.1f}s - {r.end_time:.1f}s")
            print(f"    Confidence: {r.confidence:.0f}%")
            print(f"    Modalities: {', '.join(r.modalities)}")
            if r.matched_identities:
                print(f"    Identities: {', '.join(r.matched_identities)}")
            if r.dense_context:
                ctx = r.dense_context[:100] + "..." if len(r.dense_context) > 100 else r.dense_context
                print(f"    Context: {ctx}")
            if r.vlm_reason:
                print(f"    VLM: {r.vlm_reason}")
            print()
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Search the media index")
    parser.add_argument("query", nargs="?", help="Natural language search query")
    parser.add_argument("--rerank", action="store_true", help="Enable VLM reranking")
    parser.add_argument("--limit", type=int, default=10, help="Max results")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    if not args.query:
        print('Usage: python search_cli.py "your search query" [--rerank] [--limit N] [--json]')
        sys.exit(1)
    
    asyncio.run(run_search(args.query, args.rerank, args.limit, args.json))


if __name__ == "__main__":
    main()
