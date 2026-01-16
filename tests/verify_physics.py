import asyncio
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


async def test_depth_estimator():
    logger.info("Testing DepthEstimator import...")
    from core.processing.depth_estimation import DepthEstimator
    import numpy as np

    estimator = DepthEstimator()
    assert estimator.model_name == "depth-anything-v2-small"

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    logger.info("DepthEstimator instantiated (lazy load on first use)")
    logger.success("DepthEstimator import OK")


async def test_speed_estimator():
    logger.info("Testing SpeedEstimator import...")
    from core.processing.speed_estimation import SpeedEstimator
    import numpy as np

    estimator = SpeedEstimator()
    assert estimator.model_name == "raft-small"

    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2[30:70, 30:70] = 255

    logger.info("SpeedEstimator instantiated (lazy load on first use)")
    logger.success("SpeedEstimator import OK")


async def test_clothing_detector():
    logger.info("Testing ClothingAttributeDetector import...")
    from core.processing.clothing_attributes import (
        ClothingAttributeDetector,
        CLOTHING_COLORS,
        CLOTHING_ITEMS,
        CLOTHING_PATTERNS,
    )
    import numpy as np

    detector = ClothingAttributeDetector()
    assert detector.model_name == "clip"

    assert len(CLOTHING_COLORS) >= 10
    assert len(CLOTHING_ITEMS) >= 15
    assert len(CLOTHING_PATTERNS) >= 5

    logger.info(f"Colors: {len(CLOTHING_COLORS)}, Items: {len(CLOTHING_ITEMS)}, Patterns: {len(CLOTHING_PATTERNS)}")
    logger.success("ClothingAttributeDetector import OK")


async def test_prompt_file():
    logger.info("Testing prompt file loading...")
    from pathlib import Path

    prompt_file = Path(__file__).parent.parent / "prompts" / "hyper_granular_decomposition.txt"
    assert prompt_file.exists(), f"Prompt file not found: {prompt_file}"

    content = prompt_file.read_text(encoding="utf-8")
    assert "identities" in content
    assert "clothing" in content
    assert "audio" in content
    assert "temporal" in content

    logger.info(f"Prompt file size: {len(content)} chars")
    logger.success("Prompt file loaded OK")


async def test_integration():
    logger.info("Testing HyperGranularSearcher integration with physics modules...")
    from core.retrieval.hyper_granular_search import HyperGranularSearcher

    searcher = HyperGranularSearcher()

    query = "person 2 meters away, moving fast, wearing blue jacket"
    result = await searcher.decompose_query(query)

    logger.info(f"Query: {query}")
    logger.info(f"Constraints: {result.constraint_count()}")
    logger.info(f"Clothing: {[c.model_dump() for c in result.clothing]}")
    logger.info(f"Spatial: {[s.model_dump() for s in result.spatial]}")
    logger.info(f"Actions: {[a.model_dump() for a in result.actions]}")

    assert result.constraint_count() >= 1
    logger.success("Integration test OK")


async def main():
    logger.info("=" * 60)
    logger.info("PHASE 2b VERIFICATION: Physics & Attribute Modules")
    logger.info("=" * 60)

    failed = []

    try:
        await test_depth_estimator()
    except Exception as e:
        logger.error(f"DEPTH ESTIMATOR FAILED: {e}")
        failed.append("depth_estimator")

    try:
        await test_speed_estimator()
    except Exception as e:
        logger.error(f"SPEED ESTIMATOR FAILED: {e}")
        failed.append("speed_estimator")

    try:
        await test_clothing_detector()
    except Exception as e:
        logger.error(f"CLOTHING DETECTOR FAILED: {e}")
        failed.append("clothing_detector")

    try:
        await test_prompt_file()
    except Exception as e:
        logger.error(f"PROMPT FILE FAILED: {e}")
        failed.append("prompt_file")

    try:
        await test_integration()
    except Exception as e:
        logger.error(f"INTEGRATION FAILED: {e}")
        failed.append("integration")

    logger.info("=" * 60)
    if failed:
        logger.error(f"FAILED: {failed}")
        sys.exit(1)
    else:
        logger.success("ALL PHASE 2b PHYSICS TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
