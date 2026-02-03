import asyncio
import sys
from pathlib import Path

from qdrant_client.http import models

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)


async def main():
    """Run diagnostics on the search pipeline."""
    from config import settings
    from core.storage.db import VectorDB

    print("=" * 60)
    print("AI-Media-Indexer Search Pipeline Diagnostics")
    print("=" * 60)

    # 1. Connect to database
    print("\n[1] CONNECTING TO DATABASE...")
    db = VectorDB(
        backend=settings.qdrant_backend,
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    print(f"Backend: {settings.qdrant_backend}")
    print(f"Host: {settings.qdrant_host}:{settings.qdrant_port}")

    # 2. Check collections
    print("\n[2] CHECKING COLLECTIONS...")
    try:
        collections = db.list_collections()
        print(f"Collections: {collections}")
    except Exception as e:
        print(f"Error listing collections: {e}")
        collections = []

    # 3. Count points in each collection
    print("\n[3] COLLECTION STATISTICS...")
    for coll_name in [
        "media_frames",
        "media_segments",
        "faces",
        "voices",
        "scenes",
        "scenelets",
        "masklets",
    ]:
        try:
            count = db.client.count(collection_name=coll_name)
            print(f"  {coll_name}: {count.count} points")
        except Exception:
            print(f"  {coll_name}: NOT FOUND or empty")

    # 4. Check named identities (ROBUST)
    print("\n[4] NAMED IDENTITIES (HITL)...")
    try:
        # Scan faces collection for any points with 'name' in payload
        resp = db.client.scroll(
            collection_name="faces",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="name", match=models.MatchValue(value="Alia")
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if resp[0]:
            print(f"  ✓ Found 'Alia' in faces: {resp[0][0].payload}")
        else:
            print("  ✗ 'Alia' NOT found in faces collection")

        # Generic check for any named faces
        resp = db.client.scroll(
            collection_name="faces", limit=10, with_payload=True
        )
        named_count = 0
        for point in resp[0]:
            if point.payload and point.payload.get("name"):
                print(
                    f"  Found named face: {point.payload.get('name')} (Cluster {point.payload.get('cluster_id')})"
                )
                named_count += 1
        if named_count == 0:
            print(
                "  ⚠️ No named faces found in sample of 10. You might need to label faces."
            )

    except Exception as e:
        print(f"  Error checking identifies: {e}")

    # 4b. Check SAM3 / Masklets
    print("\n[4b] SAM3 / MASKLETS...")
    try:
        mask_count = db.client.count(collection_name="masklets").count
        print(f"  Masklets count: {mask_count}")
        if mask_count > 0:
            resp = db.client.scroll(
                collection_name="masklets", limit=1, with_payload=True
            )
            print(f"  Sample masklet: {resp[0][0].payload}")
        else:
            print("  ⚠️ No masklets found. SAM3 might be disabled or failing.")
    except Exception as e:
        print(f"  Error checking masklets: {e}")

    # 5. Sample frame data
    print("\n[5] SAMPLE FRAME DATA...")
    try:
        frames = db.get_recent_frames_search(limit=3)
        for f in frames:
            print(f"\n  Frame at {f.get('timestamp', 0):.2f}s:")
            print(f"    Video: {f.get('video_path', 'unknown')}")
            # Check raw VLM output if available in payload
            print(
                f"    Description: {str(f.get('description', f.get('visual_summary', '')))[:100]}..."
            )
            print(f"    Face clusters: {f.get('face_cluster_ids', [])}")
            print(f"    Clothing colors: {f.get('clothing_colors', [])}")
            print(f"    Clothing types: {f.get('clothing_types', [])}")
            print(f"    Accessories: {f.get('accessories', [])}")
    except Exception as e:
        print(f"  Error getting frames: {e}")

    # 6. Sample scene data
    print("\n[6] SAMPLE SCENE DATA...")
    try:
        resp = db.client.scroll(
            collection_name="scenes",
            limit=3,
            with_payload=True,
        )
        if resp[0]:
            for point in resp[0]:
                payload = point.payload or {}
                print(
                    f"\n  Scene {payload.get('start_time', 0):.2f}-{payload.get('end_time', 0):.2f}s:"
                )
                print(
                    f"    Visual: {str(payload.get('visual_summary', ''))[:100]}..."
                )
                print(f"    Persons: {payload.get('person_names', [])}")
                print(
                    f"    Clothing: {payload.get('clothing_colors', [])}, {payload.get('clothing_types', [])}"
                )
                print(f"    Actions: {payload.get('actions', [])}")
        else:
            print("  No scenes found!")
            print("  ⚠️ Scene-level search won't work without scenes!")
    except Exception as e:
        print(f"  Error getting scenes: {e}")

    # 7. Test query parsing
    print("\n[7] TESTING QUERY PARSING...")
    test_query = "Alia wearing yellow skirt and ranbir wearing a white inner shirt with red outer shirt and wearing mala"
    print(f"  Query: {test_query[:80]}...")

    try:
        from core.retrieval.agentic_search import SearchAgent
        from llm.factory import LLMFactory

        llm = LLMFactory.create_llm()
        agent = SearchAgent(db=db, llm=llm)

        print("  Parsing with LLM...")
        parsed = await agent.parse_query(test_query)
        print("  ✓ Parsed successfully!")

        # Helper to safely extract names whether it uses legacy or new schema
        names = []
        if hasattr(parsed, "people") and parsed.people:
            names = [p.name for p in parsed.people]
        elif hasattr(parsed, "person_name") and parsed.person_name:
            names = [parsed.person_name]

        print(f"    Person names: {names}")
        print(f"    Entities: {len(parsed.entities)}")
        print(f"    Visual keywords: {parsed.visual_keywords[:5]}...")
        print(f"    Search text: {parsed.to_search_text()[:100]}...")

        # 9. Test basic search (NO RERANK)
        print("\n[9] SEARCH COMPARISON: NO RERANKING")
        print("  Running sota_search(use_reranking=False)...")
        results_no_rerank = await agent.sota_search(
            test_query, limit=3, use_reranking=False
        )
        print(f"  Found {len(results_no_rerank.get('results', []))} results")
        for r in results_no_rerank.get("results", []):
            print(
                f"    - Score {r.get('score', 0):.3f}: {r.get('reasoning', '')[:100]}..."
            )

        # 10. Test search (WITH RERANK)
        print("\n[10] SEARCH COMPARISON: WITH RERANKING")
        print("  Running sota_search(use_reranking=True)...")
        results_rerank = await agent.sota_search(
            test_query, limit=3, use_reranking=True
        )
        print(f"  Found {len(results_rerank.get('results', []))} results")
        for r in results_rerank.get("results", []):
            print(
                f"    - Score {r.get('combined_score', r.get('score', 0)):.3f}: {r.get('llm_reasoning', '')[:100]}..."
            )

    except Exception as e:
        print(f"  ✗ Search test failed: {e}")
        import traceback

        traceback.print_exc()

    # 11. Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    # ... (rest of summary logic)

    # 11. Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    issues = []

    # Check if scenes exist
    try:
        scene_count = db.client.count(collection_name="scenes")
        if scene_count.count == 0:
            issues.append("❌ No scenes indexed - scene search will fail")
    except:
        issues.append("❌ Scenes collection doesn't exist")

    # Check if names are registered
    try:
        named = db.get_named_face_clusters()
        if not named:
            issues.append(
                "❌ No named identities - identity resolution won't work"
            )
        else:
            names_lower = [n.lower() for _, n in named]
            if "alia" not in names_lower:
                issues.append("❌ 'Alia' not in HITL database")
            if "ranbir" not in names_lower:
                issues.append("❌ 'Ranbir' not in HITL database")
    except:
        issues.append("❌ Cannot check named identities")

    # Check if clothing info is captured
    try:
        resp = db.client.scroll(
            collection_name="media_frames",
            limit=10,
            with_payload=True,
        )
        any_clothing = False
        for point in resp[0]:
            if point.payload and (
                point.payload.get("clothing_colors")
                or point.payload.get("clothing_types")
            ):
                any_clothing = True
                break
        if not any_clothing:
            issues.append(
                "❌ No clothing info in frames - clothing filters won't work"
            )
    except:
        pass

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No obvious configuration issues found")
        print("  The problem may be in VLM quality or LLM reranking")

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
