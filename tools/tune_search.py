"""Search Quality Tuning Script.

Runs queries and displays ranked results with explainability.
Allows tuning of window_size and hybrid search alpha.

Usage:
    python tools/tune_search.py "Prakash bowling"
    python tools/tune_search.py "red shoe left foot" --alpha 0.3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage.db import VectorDB
from core.retrieval.agentic_search import SearchAgent


def run_search(query: str, limit: int = 10, use_rerank: bool = False):
    """Run search and return detailed results."""
    db = VectorDB(backend="docker", host="localhost", port=6333)
    agent = SearchAgent(db=db)
    
    print(f"\n🔍 Query: '{query}'")
    print("=" * 60)
    
    # Parse query
    parsed = agent.parse_query(query)
    print(f"\n📋 Parsed Query:")
    print(f"   Persons: {parsed.persons if hasattr(parsed, 'persons') else 'N/A'}")
    print(f"   Keywords: {parsed.expanded_keywords[:5] if hasattr(parsed, 'expanded_keywords') else 'N/A'}")
    
    # Run search
    import asyncio
    results = asyncio.run(agent.search(query, limit=limit, use_expansion=True))
    
    print(f"\n📊 Results: {len(results.get('results', []))} matches")
    print("-" * 60)
    
    for i, r in enumerate(results.get("results", [])[:limit], 1):
        score = r.get("score", 0)
        base = r.get("base_score", score)
        boost = r.get("keyword_boost", 0)
        desc = (r.get("action") or r.get("description") or "")[:80]
        video = Path(r.get("video_path", "")).name
        ts = r.get("timestamp", 0)
        
        # Identity info
        faces = r.get("face_names", [])
        speakers = r.get("speaker_names", [])
        
        print(f"\n  #{i} | Score: {score:.3f} (base={base:.3f} +boost={boost:.3f})")
        print(f"      📹 {video} @ {ts:.1f}s")
        if faces:
            print(f"      👤 Faces: {', '.join(faces)}")
        if speakers:
            print(f"      🎤 Speakers: {', '.join(speakers)}")
        print(f"      📝 {desc}...")
    
    # Output JSON report
    report = {
        "query": query,
        "parsed": str(parsed),
        "result_count": len(results.get("results", [])),
        "top_scores": [r.get("score", 0) for r in results.get("results", [])[:5]],
        "identity_matches": sum(1 for r in results.get("results", []) if r.get("face_names")),
    }
    
    report_path = Path("tests/tune_search_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n✅ Report saved to {report_path}")
    
    return results


def compare_queries(queries: list[str]):
    """Compare multiple queries side-by-side."""
    db = VectorDB(backend="docker", host="localhost", port=6333)
    
    print("\n📊 Query Comparison")
    print("=" * 60)
    
    for q in queries:
        vec = db.encode_texts(q, is_query=True)[0]
        results = db.search_frames(q, limit=3)
        top_score = results[0]["score"] if results else 0
        
        print(f"\n  '{q}'")
        print(f"    Vector norm: {sum(v**2 for v in vec)**0.5:.4f}")
        print(f"    Top score: {top_score:.4f}")
        print(f"    Results: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Quality Tuning")
    parser.add_argument("query", nargs="?", default="person walking", help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Max results")
    parser.add_argument("--compare", nargs="+", help="Compare multiple queries")
    parser.add_argument("--rerank", action="store_true", help="Enable LLM reranking")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_queries(args.compare)
    else:
        run_search(args.query, limit=args.limit, use_rerank=args.rerank)
