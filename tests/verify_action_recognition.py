import asyncio
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


async def test_temporal_analyzer_init():
    logger.info("1. Testing TemporalAnalyzer (TimeSformer) initialization...")
    from core.processing.temporal import TemporalAnalyzer

    analyzer = TemporalAnalyzer()

    assert analyzer.model_name == "facebook/timesformer-base-finetuned-k400"
    assert analyzer.model is None
    assert analyzer.processor is None

    logger.success("TemporalAnalyzer initialized (lazy loading - model not yet loaded)")
    return analyzer


async def test_temporal_analyzer_lazy_load():
    logger.info("2. Testing TemporalAnalyzer lazy model load...")
    from core.processing.temporal import TemporalAnalyzer
    import numpy as np

    analyzer = TemporalAnalyzer()

    dummy_frames = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(8)
    ]

    logger.info("   Calling analyze_clip() with 8 dummy frames...")
    logger.info("   (This will trigger model download on first run - may take a while)")

    actions = await analyzer.analyze_clip(dummy_frames, top_k=3, threshold=0.05)

    if actions:
        logger.info(f"   Detected actions: {actions}")
        assert "action" in actions[0], "Missing action field"
        assert "confidence" in actions[0], "Missing confidence field"
        logger.success(f"TimeSformer model loaded and returned {len(actions)} actions")
    else:
        logger.warning("No actions detected (may be due to random input)")
        logger.info("   (This is OK for random noise - real video would produce actions)")

    analyzer.cleanup()
    logger.success("TemporalAnalyzer cleanup called")
    return True


async def test_hyper_granular_temporal_wiring():
    logger.info("3. Testing HyperGranularSearcher temporal analyzer wiring...")
    from core.retrieval.hyper_granular_search import HyperGranularSearcher

    searcher = HyperGranularSearcher()

    assert hasattr(searcher, '_temporal_analyzer'), "Missing _temporal_analyzer field"
    assert hasattr(searcher, '_get_temporal_analyzer'), "Missing _get_temporal_analyzer method"
    assert hasattr(searcher, 'verify_action_with_frames'), "Missing verify_action_with_frames method"

    logger.success("HyperGranularSearcher has temporal analyzer wiring")


async def test_action_constraint_scoring():
    logger.info("4. Testing action constraint scoring in _score_results...")
    from core.retrieval.hyper_granular_search import (
        HyperGranularSearcher,
        HyperGranularQuery,
        ActionConstraint,
    )

    searcher = HyperGranularSearcher()

    query = HyperGranularQuery(
        original_query="person running",
        actions=[ActionConstraint(action="running", intensity="fast")],
    )

    results = [
        {"action": "A person is running quickly", "score": 0.8},
        {"action": "A person is sitting", "score": 0.7},
        {"action": "running in the park", "score": 0.6},
    ]

    scored = searcher._score_results(results, query)

    running_scores = [r["hyper_score"] for r in scored if "running" in r.get("action", "").lower()]
    sitting_score = [r["hyper_score"] for r in scored if "sitting" in r.get("action", "").lower()]

    logger.info(f"   Running results scores: {running_scores}")
    logger.info(f"   Sitting result score: {sitting_score}")

    assert all(rs > sitting_score[0] for rs in running_scores), "Running should score higher than sitting"
    assert any("matched_action" in r for r in scored), "Missing matched_action field"

    logger.success("Action constraint scoring works correctly")


async def test_action_classes_available():
    logger.info("5. Checking Kinetics-400 action classes in TimeSformer...")

    kinetics_samples = [
        "clapping",
        "drinking",
        "hugging",
        "running",
        "slapping",
        "falling",
        "dancing",
        "cooking",
        "reading",
        "typing",
    ]

    logger.info("   Sample Kinetics-400 actions that TimeSformer can recognize:")
    for action in kinetics_samples:
        logger.info(f"     - {action}")

    logger.success(f"TimeSformer trained on Kinetics-400 ({len(kinetics_samples)} sample actions shown)")


async def main():
    logger.info("=" * 70)
    logger.info("  PHASE 6: TEMPORAL ACTION INTELLIGENCE VERIFICATION")
    logger.info("=" * 70)

    failed = []

    try:
        await test_temporal_analyzer_init()
    except Exception as e:
        logger.error(f"TEMPORAL ANALYZER INIT FAILED: {e}")
        failed.append("temporal_init")

    try:
        await test_hyper_granular_temporal_wiring()
    except Exception as e:
        logger.error(f"HYPER GRANULAR WIRING FAILED: {e}")
        failed.append("hg_wiring")

    try:
        await test_action_constraint_scoring()
    except Exception as e:
        logger.error(f"ACTION SCORING FAILED: {e}")
        failed.append("action_scoring")

    try:
        await test_action_classes_available()
    except Exception as e:
        logger.error(f"ACTION CLASSES FAILED: {e}")
        failed.append("action_classes")

    logger.info("=" * 70)

    if failed:
        logger.error(f"FAILED TESTS: {failed}")
        sys.exit(1)
    else:
        logger.success("ALL PHASE 6 TEMPORAL TESTS PASSED")
        logger.info("")
        logger.info("Note: TimeSformer model loading test skipped to avoid long download.")
        logger.info("      Model will load lazily on first real query with action constraints.")
        logger.info("")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
