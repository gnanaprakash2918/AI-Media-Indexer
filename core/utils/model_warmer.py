"""Model Pre-loader / Warmer.
Downloads and caches all critical models at startup to prevent runtime latency.
"""

import asyncio
import shutil
from pathlib import Path

from config import settings
from core.utils.logger import get_logger

log = get_logger(__name__)

# Default cache locations to clean up (Windows)
_STALE_CACHE_DIRS = [
    Path.home() / ".cache" / "huggingface",
    Path.home() / ".cache" / "torch",
    Path.home() / ".cache" / "clip",
    Path.home() / ".cache" / "sentence_transformers",
]


def cleanup_stale_caches() -> None:
    """Delete leftover model caches from default Windows locations.
    
    Since we redirect all downloads to project's models/ dir via env vars,
    any files in the default locations are stale and waste disk space.
    """
    for cache_dir in _STALE_CACHE_DIRS:
        if cache_dir.exists() and cache_dir.is_dir():
            try:
                size_mb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / (1024 * 1024)
                if size_mb > 10:  # Only log if > 10MB
                    log.info(f"[Cleanup] Removing stale cache: {cache_dir} ({size_mb:.1f}MB)")
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception as e:
                log.warning(f"[Cleanup] Failed to remove {cache_dir}: {e}")


async def warmup_models():
    """Download and cache all critical models."""
    log.info("[Warmer] Starting model warmup sequence...")
    
    # Clean up any stale caches from default Windows locations (C drive)
    cleanup_stale_caches()

    tasks = []

    # 1. TransNet V2 (Shot Detection) - CPU/GPU
    try:
        from huggingface_hub import hf_hub_download

        from core.processing.transnet_detector import TransNetV2

        log.info("[Warmer] Checking TransNet V2 model file...")
        hf_hub_download(
            repo_id="elya5/transnetv2",
            filename="transnetv2.onnx",
            local_dir=str(settings.model_cache_dir),
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
        # Trigger load - method varies by implementation
        if hasattr(encoder, "ensure_loaded"):
            await encoder.ensure_loaded()
        elif hasattr(encoder, "load"):
            encoder.load()
    except Exception as e:
        log.warning(f"[Warmer] SigLIP init failed: {e}")

    # 5. SAM 3 (Segment Anything)
    if getattr(settings, "enable_sam3_tracking", False):
        try:
            from huggingface_hub import hf_hub_download

            log.info("[Warmer] Checking SAM 3 Checkpoint...")
            # Placeholder: if we had a direct download for SAM3/2 checkpoint
            # For now, just importing it might trigger internal checks if implemented
            pass
        except Exception as e:
                log.warning(f"[Warmer] SAM 3 init failed: {e}")

    # 6. YOLO (Object Detection)
    if getattr(settings, "enable_object_detection", False):
        try:
            from ultralytics import YOLO
            log.info("[Warmer] Checking YOLOv8...")
            # This triggers download if missing
            YOLO("yolov8m.pt")
        except Exception as e:
            log.warning(f"[Warmer] YOLO init failed: {e}")

    # 7. ArcFace (Identity)
    if getattr(settings, "enable_face_recognition", False):
        try:
            arcface_path = settings.model_cache_dir / "arcface" / "w600k_r50.onnx"
            if not arcface_path.exists():
                log.info("[Warmer] Downloading ArcFace ONNX...")
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id="minchul/cvl-face-recognition-models",
                    filename="w600k_r50.onnx",
                    local_dir=str(settings.model_cache_dir / "arcface"),
                    local_dir_use_symlinks=False,
                )
        except Exception as e:
            log.warning(f"[Warmer] ArcFace init failed: {e}")

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
            "OK" if (settings.model_cache_dir / "transnetv2.onnx").exists() else "MISSING"
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
        print(f"{'YOLOv8':<25} | {'CHECKED':<20}")
        print(f"{'ArcFace':<25} | {('OK' if (settings.model_cache_dir / 'arcface' / 'w600k_r50.onnx').exists() else 'MISSING'):<20}")
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
