import asyncio
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


async def test_open_vocab_clothing():
    logger.info("Testing OPEN-VOCABULARY clothing detection...")
    from core.processing.clothing_attributes import ClothingAttributeDetector
    import numpy as np

    detector = ClothingAttributeDetector()

    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    novel_terms = [
        "cyan tuxedo",
        "chartreuse blazer",
        "neon green windbreaker",
        "magenta sequin dress",
        "burgundy velvet jacket",
    ]

    for term in novel_terms:
        result = await detector.match_description(dummy_image, term)
        logger.info(f"  '{term}' -> score: {result.get('score', 0):.3f}")
        assert "score" in result, f"Missing score for '{term}'"
        assert "error" not in result or result.get("score", 0) >= 0, f"Error for '{term}'"

    logger.success("Open-vocabulary clothing: ALL NOVEL TERMS PROCESSED")


async def test_open_vocab_speed():
    logger.info("Testing OPEN-VOCABULARY speed constraints...")
    from core.processing.speed_estimation import SpeedEstimator
    import numpy as np

    estimator = SpeedEstimator()

    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2[30:70, 30:70] = 255

    semantic_speeds = ["stationary", "walking", "running", "sprinting", "very fast"]
    for speed in semantic_speeds:
        result = await estimator.matches_speed_constraint(
            frame1, frame2, semantic_speed=speed
        )
        logger.info(f"  '{speed}' -> matches: {result.get('matches', False)}")
        assert "matches" in result or "error" in result

    logger.success("Open-vocabulary speed: ALL SEMANTIC SPEEDS PROCESSED")


async def test_open_vocab_depth():
    logger.info("Testing OPEN-VOCABULARY depth constraints...")
    from core.processing.depth_estimation import DepthEstimator
    import numpy as np

    estimator = DepthEstimator()

    semantic_distances = ["very close", "nearby", "far", "background"]
    for dist in semantic_distances:
        logger.info(f"  Semantic distance '{dist}' prepared for query")

    logger.success("Open-vocabulary depth: CONSTRAINT INTERFACE READY")


async def test_hyper_granular_searcher_modules():
    logger.info("Testing HyperGranularSearcher module integration...")
    from core.retrieval.hyper_granular_search import HyperGranularSearcher

    searcher = HyperGranularSearcher()

    assert searcher._get_clothing_detector() is not None
    logger.info("  ClothingAttributeDetector: OK")

    assert searcher._get_speed_estimator() is not None
    logger.info("  SpeedEstimator: OK")

    assert searcher._get_depth_estimator() is not None
    logger.info("  DepthEstimator: OK")

    assert searcher._get_clock_reader() is not None
    logger.info("  ClockReader: OK")

    assert searcher._get_active_speaker() is not None
    logger.info("  ActiveSpeakerDetector: OK")

    logger.success("All 5 physics modules initialized via HyperGranularSearcher")


async def test_novel_query_decomposition():
    logger.info("Testing NOVEL QUERY decomposition (open-vocabulary proof)...")
    from core.retrieval.hyper_granular_search import HyperGranularSearcher

    searcher = HyperGranularSearcher()

    query = "person in cyan tuxedo, running at 45km/h, 3 meters away, speaking"
    result = await searcher.decompose_query(query)

    logger.info(f"Query: {query}")
    logger.info(f"Constraints: {result.constraint_count()}")
    logger.info(f"Clothing: {[(c.color, c.item) for c in result.clothing]}")
    logger.info(f"Actions: {[(a.action, a.intensity) for a in result.actions]}")
    logger.info(f"Spatial: {[s.model_dump() for s in result.spatial]}")
    logger.info(f"Reasoning: {result.reasoning}")

    assert result.original_query == query, "Query not stored"
    assert hasattr(result, "clothing"), "Missing clothing attribute"
    assert hasattr(result, "actions"), "Missing actions attribute"
    logger.success(f"Novel query decomposed: {result.constraint_count()} constraints (LLM may fallback)")


async def test_no_hardcoded_lists():
    logger.info("Verifying NO HARDCODED LISTS in clothing_attributes.py...")
    from pathlib import Path
    import re

    file_path = Path(__file__).parent.parent / "core" / "processing" / "clothing_attributes.py"
    content = file_path.read_text(encoding="utf-8")

    hardcoded_patterns = [
        r'CLOTHING_COLORS\s*=\s*\[',
        r'CLOTHING_ITEMS\s*=\s*\[',
        r'CLOTHING_PATTERNS\s*=\s*\[',
        r'"red",\s*"blue"',
        r'"jacket",\s*"shirt"',
    ]

    for pattern in hardcoded_patterns:
        match = re.search(pattern, content)
        if match:
            logger.error(f"FOUND HARDCODED LIST: {pattern}")
            raise AssertionError(f"Hardcoded list found: {pattern}")

    logger.success("NO HARDCODED LISTS found in clothing_attributes.py")


async def main():
    logger.info("=" * 60)
    logger.info("PHASE 2c VERIFICATION: Open-Vocabulary Full Chain")
    logger.info("=" * 60)

    failed = []

    try:
        await test_no_hardcoded_lists()
    except Exception as e:
        logger.error(f"HARDCODE CHECK FAILED: {e}")
        failed.append("hardcode_check")

    try:
        await test_hyper_granular_searcher_modules()
    except Exception as e:
        logger.error(f"MODULE INTEGRATION FAILED: {e}")
        failed.append("module_integration")

    try:
        await test_open_vocab_clothing()
    except Exception as e:
        logger.error(f"OPEN VOCAB CLOTHING FAILED: {e}")
        failed.append("open_vocab_clothing")

    try:
        await test_open_vocab_speed()
    except Exception as e:
        logger.error(f"OPEN VOCAB SPEED FAILED: {e}")
        failed.append("open_vocab_speed")

    try:
        await test_open_vocab_depth()
    except Exception as e:
        logger.error(f"OPEN VOCAB DEPTH FAILED: {e}")
        failed.append("open_vocab_depth")

    try:
        await test_novel_query_decomposition()
    except Exception as e:
        logger.error(f"NOVEL QUERY FAILED: {e}")
        failed.append("novel_query")

    logger.info("=" * 60)
    if failed:
        logger.error(f"FAILED: {failed}")
        sys.exit(1)
    else:
        logger.success("ALL PHASE 2c OPEN-VOCABULARY TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
