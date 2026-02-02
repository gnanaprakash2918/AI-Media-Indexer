import asyncio
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test_sota")


async def test_deep_research():
    print("--- Testing Deep Research Integration ---")

    try:
        from core.processing.analysis.deep_research import get_deep_research_processor

        processor = get_deep_research_processor()
        print("[OK] Processor initialized")

        # Test lazy loaders (just check if they *can* load, don't necessarily download huge models if not needed)
        # But we want to verifiable correct wiring.

        # Create a dummy black frame
        import numpy as np
        import cv2

        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        print("Running analyze_frame on dummy black frame...")
        result = await processor.analyze_frame(
            frame=dummy_frame,
            compute_aesthetics=True,
            compute_saliency=False,  # Skip heavy
            compute_fingerprint=True,
        )

        print("\n[Result]")
        print(f"Shot Type: {result.shot_type}")
        print(f"Mood: {result.mood}")
        print(f"Aesthetic Score: {result.aesthetic_score}")
        print(f"Is Black Frame: {result.is_black_frame}")

        if result.is_black_frame:
            print("[Pass] Correctly identified black frame")
        else:
            print("[Warn] technical_detector check failed or disabled")

        print("\n--- Test Complete ---")

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_deep_research())
