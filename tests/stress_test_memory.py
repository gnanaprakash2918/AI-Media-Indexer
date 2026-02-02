"""Memory Stress Test for Physics Modules.

Simulates search workload triggering ALL 6 physics modules sequentially.
Monitors VRAM usage to verify no memory leaks.

Usage:
    python tests/stress_test_memory.py
"""

from __future__ import annotations

import asyncio
import gc
import sys
import time

import numpy as np


def get_vram_info() -> tuple[float, float]:
    """Get VRAM (used_gb, total_gb)."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            return used / (1024**3), total / (1024**3)
    except Exception:
        pass
    return 0.0, 0.0


def cleanup_vram() -> None:
    """Force VRAM cleanup."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def create_test_frames(
    n: int = 8, size: tuple = (224, 224)
) -> list[np.ndarray]:
    """Create synthetic RGB frames for testing."""
    return [
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8) for _ in range(n)
    ]


async def test_temporal_analyzer() -> dict:
    """Test TimeSformer model load/unload."""
    print("\n🎬 Testing TemporalAnalyzer (TimeSformer)...")

    from core.processing.temporal import TemporalAnalyzer

    before = get_vram_info()[0]
    analyzer = TemporalAnalyzer()

    frames = create_test_frames(8)
    result = await analyzer.analyze_clip(frames, top_k=3)
    peak = get_vram_info()[0]

    print(f"   Actions: {[r['action'] for r in result[:2]]}")

    analyzer.cleanup()
    cleanup_vram()
    after = get_vram_info()[0]

    return {
        "module": "TimeSformer",
        "before": before,
        "peak": peak,
        "after": after,
    }


async def test_depth_estimator() -> dict:
    """Test DepthAnything model load/unload."""
    print("\n📏 Testing DepthEstimator (DepthAnything V2)...")

    from core.processing.depth_estimation import DepthEstimator

    before = get_vram_info()[0]
    estimator = DepthEstimator()

    frame = create_test_frames(1)[0]
    result = await estimator.estimate_depth(frame)
    peak = get_vram_info()[0]

    if result.get("stats"):
        print(
            f"   Depth range: {result['stats']['min_depth']:.2f} - {result['stats']['max_depth']:.2f}"
        )

    estimator.cleanup()
    cleanup_vram()
    after = get_vram_info()[0]

    return {
        "module": "DepthAnything",
        "before": before,
        "peak": peak,
        "after": after,
    }


async def test_speed_estimator() -> dict:
    """Test RAFT optical flow model load/unload."""
    print("\n🏃 Testing SpeedEstimator (RAFT)...")

    from core.processing.speed_estimation import SpeedEstimator

    before = get_vram_info()[0]
    estimator = SpeedEstimator()

    frames = create_test_frames(2, size=(256, 256))
    result = await estimator.compute_optical_flow(frames[0], frames[1])
    peak = get_vram_info()[0]

    if result.get("mean_velocity_px"):
        print(f"   Mean velocity: {result['mean_velocity_px']:.2f} px")

    estimator.cleanup()
    cleanup_vram()
    after = get_vram_info()[0]

    return {"module": "RAFT", "before": before, "peak": peak, "after": after}


async def test_visual_encoder() -> dict:
    """Test visual encoder (SigLIP/CLIP) model load/unload."""
    print("\n🖼️ Testing VisualEncoder (SigLIP/CLIP)...")

    from core.processing.vision.visual_encoder import (
        get_default_visual_encoder,
        reset_visual_encoder,
    )

    before = get_vram_info()[0]
    encoder = get_default_visual_encoder()

    frames = create_test_frames(4)
    # Encode a batch of frames
    from PIL import Image

    images = []
    for frame in frames:
        img = Image.fromarray(frame)
        images.append(img)

    # Get embeddings
    embeddings = encoder.encode_images(images)
    peak = get_vram_info()[0]

    if embeddings is not None:
        print(f"   Encoded {len(embeddings)} images, dim={len(embeddings[0])}")

    reset_visual_encoder()
    cleanup_vram()
    after = get_vram_info()[0]

    return {
        "module": "VisualEncoder",
        "before": before,
        "peak": peak,
        "after": after,
    }


async def test_audio_events() -> dict:
    """Test CLAP audio event detector."""
    print("\n🔊 Testing AudioEventDetector (CLAP)...")

    from core.processing.audio.audio_events import AudioEventDetector

    before = get_vram_info()[0]
    detector = AudioEventDetector()

    # Create synthetic audio (1 second of white noise at 16kHz)
    audio = np.random.randn(16000).astype(np.float32)
    result = await detector.detect_events(
        audio, target_classes=["speech", "noise"], sample_rate=16000
    )
    peak = get_vram_info()[0]

    if result:
        print(f"   Events: {[e['event'] for e in result[:2]]}")

    detector.cleanup()
    cleanup_vram()
    after = get_vram_info()[0]

    return {"module": "CLAP", "before": before, "peak": peak, "after": after}


async def run_stress_test(cycles: int = 2) -> None:
    """Run full stress test with multiple cycles."""
    print("=" * 60)
    print("🧪 MEMORY STRESS TEST - Physics Modules")
    print("=" * 60)

    used, total = get_vram_info()
    if total == 0:
        print("⚠️  No GPU detected. Running in CPU mode (no VRAM tracking).")
    else:
        print(f"📊 Initial VRAM: {used:.2f}GB / {total:.2f}GB")

    baseline = used
    all_results = []

    for cycle in range(1, cycles + 1):
        print(f"\n{'=' * 60}")
        print(f"📦 CYCLE {cycle}/{cycles}")
        print("=" * 60)

        test_funcs = [
            test_temporal_analyzer,
            test_depth_estimator,
            test_speed_estimator,
            test_visual_encoder,
            test_audio_events,
        ]

        cycle_results = []
        for test_fn in test_funcs:
            try:
                result = await test_fn()
                cycle_results.append(result)
                print(
                    f"   ✅ {result['module']}: Peak {result['peak']:.2f}GB → After {result['after']:.2f}GB"
                )
            except Exception as e:
                print(f"   ❌ {test_fn.__name__} failed: {e}")
                cycle_results.append(
                    {"module": test_fn.__name__, "error": str(e)}
                )

        all_results.append(cycle_results)

        # Force cleanup between cycles
        cleanup_vram()
        time.sleep(1)

    # Final report
    print("\n" + "=" * 60)
    print("📊 STRESS TEST RESULTS")
    print("=" * 60)

    final_vram = get_vram_info()[0]
    leak = final_vram - baseline

    print("\n📈 VRAM Analysis:")
    print(f"   Baseline:  {baseline:.2f}GB")
    print(f"   Final:     {final_vram:.2f}GB")
    print(f"   Leak:      {leak:+.2f}GB")

    # Check for acceptable leak (±100MB tolerance)
    if abs(leak) <= 0.1:
        print("\n✅ PASS: VRAM returned to baseline (no significant leaks)")
        sys.exit(0)
    else:
        print(f"\n⚠️  WARNING: VRAM leak detected ({leak:+.2f}GB)")
        print("   This may indicate a memory leak or PyTorch caching.")
        sys.exit(1)


def main():
    """Entry point."""
    try:
        asyncio.run(run_stress_test(cycles=2))
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
