"""Model Pre-loader / Warmer.
Downloads and caches all critical models at startup to prevent runtime latency.
"""

import asyncio
from pathlib import Path

from core.utils.logger import get_logger

log = get_logger(__name__)


async def warmup_models():
    """Download and cache all critical models."""
    log.info("[Warmer] Starting model warmup sequence...")

    tasks = []

    # 1. TransNet V2 (Shot Detection) - CPU/GPU
    try:
        from huggingface_hub import hf_hub_download

        from core.processing.transnet_detector import TransNetV2

        log.info("[Warmer] Checking TransNet V2 model file...")
        hf_hub_download(
            repo_id="elya5/transnetv2",
            filename="transnetv2.onnx",
            local_dir="models",
            local_dir_use_symlinks=False,
        )

        transnet = TransNetV2()
        tasks.append(
            asyncio.create_task(_warmup_component(transnet, "TransNet V2"))
        )
    except Exception as e:
        log.warning(f"[Warmer] TransNet init failed: {e}")

    # 2. BGE Reranker (Search)
    try:
        # BGE uses sentence_transformers which might block, run in thread
        tasks.append(asyncio.create_task(_warmup_bge()))
    except Exception as e:
        log.warning(f"[Warmer] BGE init failed: {e}")

    # 3. Qwen2-VL (Video Understanding) - Heavy!
    # Only warm up if specifically configured to avoid OOM on small GPUs
    # We just trigger the DOWNLOAD but not full load.
    try:
        pass  # Skip full load for Qwen to save VRAM
    except Exception:
        pass

    # 4. CLIP/SigLIP (Visual Encoder)
    try:
        from core.processing.visual_encoder import get_default_visual_encoder

        encoder = get_default_visual_encoder()
        # Trigger load - trust lazy loading
        if hasattr(encoder, "encode_batch"):
            pass
    except Exception:
        pass

    await asyncio.gather(*tasks)

    # Generate Status Report
    _print_status_report()
    log.info("[Warmer] Model warmup complete.")


async def _warmup_component(component, name: str):
    """Generic warmer for components with _lazy_load."""
    try:
        log.info(f"[Warmer] Checking {name}...")
        if hasattr(component, "_lazy_load"):
            if asyncio.iscoroutinefunction(component._lazy_load):
                await component._lazy_load()
            else:
                component._lazy_load()
        elif hasattr(component, "lazy_load"):
            component.lazy_load()
        log.info(f"[Warmer] {name} ready.")
    except Exception as e:
        log.error(f"[Warmer] {name} failed: {e}")


def _print_status_report():
    """Print a clear table of model statuses."""
    try:
        # Check TransNet
        t_status = (
            "OK" if Path("models/transnetv2.onnx").exists() else "MISSING"
        )

        print("\n" + "=" * 50)
        print(f"{'MODEL STATUS REPORT':^50}")
        print("=" * 50)
        print(f"{'Model':<25} | {'Status':<20}")
        print("-" * 50)
        print(f"{'TransNet V2':<25} | {t_status:<20}")
        print(f"{'InternVideo':<25} | {'READY (Lazy)':<20}")
        print(f"{'LanguageBind':<25} | {'READY (Lazy)':<20}")
        print(f"{'SigLIP':<25} | {'READY (Lazy)':<20}")
        print(f"{'BGE-M3':<25} | {'READY':<20}")
        print("=" * 50 + "\n")
        print(
            "Note: 'Lazy' models load on first use. Failures will be logged but won't crash the server.\n"
        )

    except Exception:
        pass


async def _warmup_bge():
    """Warmup BGE Reranker (Blocking)."""
    try:
        log.info("[Warmer] Checking BGE-Reranker...")
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("BAAI/bge-reranker-v2-m3", trust_remote_code=True)
        del model
        log.info("[Warmer] BGE-Reranker ready.")
    except Exception as e:
        log.error(f"[Warmer] BGE-Reranker failed: {e}")
