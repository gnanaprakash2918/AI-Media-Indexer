import asyncio
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


async def test_schema_imports():
    logger.info("Testing schema imports...")
    from core.retrieval.hyper_granular_search import (
        IdentityConstraint,
        ClothingConstraint,
        AudioConstraint,
        TemporalConstraint,
        TextConstraint,
        SpatialConstraint,
        ActionConstraint,
        HyperGranularQuery,
    )

    ic = IdentityConstraint(name="Prakash")
    assert ic.name == "Prakash"

    cc = ClothingConstraint(body_part="upper", color="blue", item="t-shirt")
    assert cc.color == "blue"
    assert cc.item == "t-shirt"

    ac = AudioConstraint(event_class="cheer", min_db=92)
    assert ac.event_class == "cheer"
    assert ac.min_db == 92

    tc = TemporalConstraint(constraint_type="delay", min_ms=500, max_ms=2000)
    assert tc.min_ms == 500

    txt = TextConstraint(text="Brunswick Sports", location="sign")
    assert txt.text == "Brunswick Sports"

    sp = SpatialConstraint(measurement_type="distance", value_cm=150.0)
    assert sp.value_cm == 150.0

    act = ActionConstraint(action="bowling", result="strike")
    assert act.action == "bowling"

    logger.success("All schema imports and instantiation OK")


async def test_fallback_decomposition():
    logger.info("Testing fallback decomposition (no LLM)...")
    from core.retrieval.hyper_granular_search import HyperGranularSearcher

    searcher = HyperGranularSearcher()
    query = "Prakash blue t-shirt, 92dB cheer, 500ms delay"
    result = await searcher.decompose_query(query)

    logger.info(f"Original query: {result.original_query}")
    logger.info(f"Total constraints: {result.constraint_count()}")
    logger.info(f"Identities: {[i.model_dump() for i in result.identities]}")
    logger.info(f"Clothing: {[c.model_dump() for c in result.clothing]}")
    logger.info(f"Audio: {[a.model_dump() for a in result.audio]}")
    logger.info(f"Temporal: {[t.model_dump() for t in result.temporal]}")
    logger.info(f"Reasoning: {result.reasoning}")

    assert result.original_query == query
    assert result.constraint_count() > 0
    logger.success(f"Fallback decomposition: {result.constraint_count()} constraints extracted")

    return result


async def test_json_serialization():
    logger.info("Testing JSON serialization...")
    from core.retrieval.hyper_granular_search import HyperGranularSearcher

    searcher = HyperGranularSearcher()
    query = "Prakash wearing red sneaker on left foot, green sneaker on right foot"
    result = await searcher.decompose_query(query)

    json_output = result.model_dump()

    assert "original_query" in json_output
    assert "identities" in json_output
    assert "clothing" in json_output
    assert "audio" in json_output
    assert "temporal" in json_output
    assert "text" in json_output
    assert "scene_description" in json_output
    assert "reasoning" in json_output

    logger.info(f"JSON keys: {list(json_output.keys())}")
    logger.success("JSON serialization OK")


async def test_complex_query():
    logger.info("Testing complex multi-constraint query...")
    from core.retrieval.hyper_granular_search import HyperGranularSearcher

    searcher = HyperGranularSearcher()
    query = "Prakash in blue t-shirt, red sneaker left, green sneaker right, bowling strike with 500-2000ms delay, Brunswick Sports sign visible, 92dB crowd cheer"
    result = await searcher.decompose_query(query)

    logger.info("=" * 50)
    logger.info(f"COMPLEX QUERY: {query}")
    logger.info("=" * 50)
    logger.info(f"Constraints extracted: {result.constraint_count()}")
    logger.info(f"Identities: {[i.name for i in result.identities]}")
    logger.info(f"Clothing: {[(c.color, c.item, c.side) for c in result.clothing]}")
    logger.info(f"Audio: {[(a.event_class, a.min_db) for a in result.audio]}")
    logger.info(f"Temporal: {[(t.constraint_type, t.min_ms, t.max_ms) for t in result.temporal]}")
    logger.info(f"Text: {[t.text for t in result.text]}")
    logger.info(f"Scene: {result.scene_description}")
    logger.info("=" * 50)

    assert result.constraint_count() >= 1, f"Expected >= 1 constraint, got {result.constraint_count()}"
    logger.success(f"Complex query: {result.constraint_count()} constraints")


async def main():
    logger.info("=" * 60)
    logger.info("PHASE 2 VERIFICATION: Hyper-Granular Search Decomposition")
    logger.info("=" * 60)

    failed = []

    try:
        await test_schema_imports()
    except Exception as e:
        logger.error(f"SCHEMA IMPORTS FAILED: {e}")
        failed.append("schema_imports")

    try:
        await test_fallback_decomposition()
    except Exception as e:
        logger.error(f"FALLBACK DECOMPOSITION FAILED: {e}")
        failed.append("fallback_decomposition")

    try:
        await test_json_serialization()
    except Exception as e:
        logger.error(f"JSON SERIALIZATION FAILED: {e}")
        failed.append("json_serialization")

    try:
        await test_complex_query()
    except Exception as e:
        logger.error(f"COMPLEX QUERY FAILED: {e}")
        failed.append("complex_query")

    logger.info("=" * 60)
    if failed:
        logger.error(f"FAILED: {failed}")
        sys.exit(1)
    else:
        logger.success("ALL PHASE 2 DECOMPOSITION TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
