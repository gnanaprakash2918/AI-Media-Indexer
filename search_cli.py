"""CLI tool for searching the media index."""

import argparse
import asyncio
import json
import sys

from core.storage.db import VectorDB


async def run_search(
    query: str, use_rerank: bool, limit: int, as_json: bool
) -> None:
    """Executes the search and prints results."""
    # REFACTOR: Use SearchAgent instead of missing core.retrieval.engine
    from core.retrieval.agentic_search import SearchAgent
    
    # Initialize DB (using environment settings logic if possible, or default to docker/host)
    # NOTE: VectorDB backend 'docker' might need to be 'qdrant' or None depending on DB impl.
    db = VectorDB() 
    
    # Initialize Agent
    agent = SearchAgent(db=db)

    try:
        # returns dict with 'results': [dict, ...]
        result_pkg = await agent.sota_search(
            query=query, 
            use_reranking=use_rerank, 
            limit=limit,
            use_expansion=True
        )
        
        results = result_pkg.get("results", [])

        if as_json:
            print(json.dumps(results, indent=2))
            return

        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"Results: {len(results)}")
        print(f"{'=' * 60}\n")

        for i, r in enumerate(results, 1):
            # Adapt keys from agent result (dicts)
            file_path = r.get("video_path") or r.get("media_path", "N/A")
            start = r.get("start_time") or r.get("timestamp", 0.0)
            end = r.get("end_time") or r.get("end", 0.0)
            score = r.get("score", 0.0)
            
            # Reasons/Explanation
            reasons = r.get("match_reasons", [])
            if not reasons and r.get("reasoning"):
                reasons = [r.get("reasoning")]
            elif not reasons and r.get("llm_reasoning"):
                reasons = [r.get("llm_reasoning")]
                
            explanation = r.get("explanation", "")
            
            # Identity
            # 'face_names' or 'person_names' might be lists
            identities = r.get("person_names") or r.get("face_names", [])

            print(f"[{i}] {file_path}")
            print(f"    Time: {float(start):.1f}s - {float(end):.1f}s")
            print(f"    Score: {float(score) * 100:.0f}%")
            if reasons:
                print(f"    Reasons: {', '.join(reasons)}")
            if identities:
                print(f"    Identities: {', '.join(identities)}")
            
            # Context/Content
            ctx = r.get("visual_summary") or r.get("content_text") or r.get("action", "")
            if ctx:
                ctx_preview = (
                    ctx[:100] + "..."
                    if len(ctx) > 100
                    else ctx
                )
                print(f"    Context: {ctx_preview}")
            
            if explanation:
                print(f"    Explanation: {explanation}")
            print()
    except Exception as e:
        print(f"Search failed: {e}")
    finally:
        # VectorDB might not have close method if it's just a client wrapper, but checking
        if hasattr(db, "client") and hasattr(db.client, "close"):
            db.client.close()



def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Search the media index")
    parser.add_argument(
        "query", nargs="?", help="Natural language search query"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Enable VLM reranking"
    )
    parser.add_argument("--limit", type=int, default=10, help="Max results")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not args.query:
        print(
            'Usage: python search_cli.py "your search query" [--rerank] [--limit N] [--json]'
        )
        sys.exit(1)

    asyncio.run(run_search(args.query, args.rerank, args.limit, args.json))


if __name__ == "__main__":
    main()
