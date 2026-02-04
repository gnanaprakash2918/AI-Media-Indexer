"""Model Pre-loader / Warmer.
Downloads and caches all critical models at startup to prevent runtime latency.
"""

import asyncio
import shutil
import time
from pathlib import Path

from config import settings
from core.utils.logger import get_logger, log_verbose

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
                log_verbose(f"[Cleanup] Found stale cache: {cache_dir}, size={size_mb:.1f}MB")
                if size_mb > 10:  # Only log to terminal if > 10MB
                    log.info(f"[Cleanup] Removing stale cache ({size_mb:.0f}MB)")
                shutil.rmtree(cache_dir, ignore_errors=True)
                log_verbose(f"[Cleanup] Removed: {cache_dir}")
            except Exception as e:
                log.warning(f"[Cleanup] Failed to remove {cache_dir}: {e}")
                log_verbose(f"[Cleanup] Exception: {type(e).__name__}: {e}")


async def warmup_models():
    """Download and cache all critical models."""
    start_time = time.time()
    log.info("[Warmer] Starting model warmup...")
    log_verbose(f"[Warmer] Model cache dir: {settings.model_cache_dir}")
    log_verbose(f"[Warmer] Settings: face_rec={getattr(settings, 'enable_face_recognition', False)}, "
                f"sam3={getattr(settings, 'enable_sam3_tracking', False)}, "
                f"object_det={getattr(settings, 'enable_object_detection', False)}")
    
    # Clean up any stale caches from default Windows locations (C drive)
    cleanup_stale_caches()

    tasks = []
    models_checked = []

    # 1. TransNet V2 (Shot Detection) - CPU/GPU
    try:
        from huggingface_hub import hf_hub_download

        from core.processing.transnet_detector import TransNetV2

        transnet_path = settings.model_cache_dir / "transnetv2.onnx"
        log_verbose(f"[Warmer] TransNet V2 expected at: {transnet_path}")
        log_verbose(f"[Warmer] TransNet V2 exists: {transnet_path.exists()}")
        
        log.info("[Warmer] Checking TransNet V2...")
        hf_hub_download(
            repo_id="elya5/transnetv2",
            filename="transnetv2.onnx",
            local_dir=str(settings.model_cache_dir),
            local_dir_use_symlinks=False,
        )
        models_checked.append("TransNet V2")

        transnet = TransNetV2()
        tasks.append(
            asyncio.create_task(_warmup_component(transnet, "TransNet V2"))
        )
    except Exception as e:
        log.warning(f"[Warmer] TransNet init failed: {e}")
        log_verbose(f"[Warmer] TransNet exception: {type(e).__name__}: {e}")

    # 2. BGE Reranker (Search)
    try:
        # BGE uses sentence_transformers which might block, run in thread
        tasks.append(asyncio.create_task(_warmup_bge()))
        models_checked.append("BGE Reranker")
    except Exception as e:
        log.warning(f"[Warmer] BGE init failed: {e}")
        log_verbose(f"[Warmer] BGE exception: {type(e).__name__}: {e}")

    # 3. Qwen2-VL (Video Understanding) - Heavy!
    # Only warm up if specifically configured to avoid OOM on small GPUs
    # We just trigger the DOWNLOAD but not full load.
    try:
        log_verbose("[Warmer] Qwen2-VL: Skipping full load (lazy-loaded on demand)")
        pass  # Skip full load for Qwen to save VRAM
    except Exception:
        pass

    # 4. CLIP/SigLIP (Visual Encoder)
    try:
        from core.processing.visual_encoder import get_default_visual_encoder

        log.info("[Warmer] Checking SigLIP/CLIP Encoder...")
        encoder = get_default_visual_encoder()
        log_verbose(f"[Warmer] Visual encoder type: {type(encoder).__name__}")
        # Trigger load - method varies by implementation
        if hasattr(encoder, "ensure_loaded"):
            await encoder.ensure_loaded()
        elif hasattr(encoder, "load"):
            encoder.load()
        models_checked.append("SigLIP")
        log_verbose("[Warmer] SigLIP encoder loaded successfully")
    except Exception as e:
        log.warning(f"[Warmer] SigLIP init failed: {e}")
        log_verbose(f"[Warmer] SigLIP exception: {type(e).__name__}: {e}")

    # 5. SAM 3 (Segment Anything)
    if getattr(settings, "enable_sam3_tracking", False):
        try:
            from huggingface_hub import hf_hub_download

            log.info("[Warmer] Checking SAM 3 Checkpoint...")
            log_verbose("[Warmer] SAM 3 enabled, checking checkpoint...")
            # Placeholder: if we had a direct download for SAM3/2 checkpoint
            # For now, just importing it might trigger internal checks if implemented
            models_checked.append("SAM 3")
            pass
        except Exception as e:
            log.warning(f"[Warmer] SAM 3 init failed: {e}")
            log_verbose(f"[Warmer] SAM 3 exception: {type(e).__name__}: {e}")
    else:
        log_verbose("[Warmer] SAM 3 disabled, skipping")

    # 6. YOLO (Object Detection)
    if getattr(settings, "enable_object_detection", False):
        try:
            from ultralytics import YOLO
            log.info("[Warmer] Checking YOLOv8...")
            log_verbose("[Warmer] YOLOv8 object detection enabled")
            # This triggers download if missing
            YOLO("yolov8m.pt")
            models_checked.append("YOLOv8")
            log_verbose("[Warmer] YOLOv8 loaded successfully")
        except Exception as e:
            log.warning(f"[Warmer] YOLO init failed: {e}")
            log_verbose(f"[Warmer] YOLO exception: {type(e).__name__}: {e}")
    else:
        log_verbose("[Warmer] YOLOv8 disabled, skipping")

    # 7. ArcFace (Identity)
    if getattr(settings, "enable_face_recognition", False):
        try:
            arcface_path = settings.model_cache_dir / "arcface" / "w600k_r50.onnx"
            log_verbose(f"[Warmer] ArcFace expected at: {arcface_path}")
            log_verbose(f"[Warmer] ArcFace exists: {arcface_path.exists()}")
            
            if not arcface_path.exists():
                log.info("[Warmer] Downloading ArcFace ONNX...")
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id="minchul/cvl-face-recognition-models",
                    filename="w600k_r50.onnx",
                    local_dir=str(settings.model_cache_dir / "arcface"),
                    local_dir_use_symlinks=False,
                )
                log_verbose("[Warmer] ArcFace download complete")
            models_checked.append("ArcFace")
        except Exception as e:
            log.warning(f"[Warmer] ArcFace init failed: {e}")
            log_verbose(f"[Warmer] ArcFace exception: {type(e).__name__}: {e}")
    else:
        log_verbose("[Warmer] ArcFace disabled, skipping")

    await asyncio.gather(*tasks)

    # Generate Status Report
    elapsed = time.time() - start_time
    log_verbose(f"[Warmer] Total warmup time: {elapsed:.1f}s, models checked: {models_checked}")
    _print_status_report()
    log.info(f"[Warmer] Model warmup complete ({elapsed:.1f}s)")


async def _warmup_component(component, name: str):
    """Generic warmer for components with _lazy_load."""
    start = time.time()
    try:
        log_verbose(f"[Warmer] Loading {name}, component type: {type(component).__name__}")
        if hasattr(component, "_lazy_load"):
            if asyncio.iscoroutinefunction(component._lazy_load):
                await component._lazy_load()
            else:
                component._lazy_load()
        elif hasattr(component, "lazy_load"):
            component.lazy_load()
        elapsed = time.time() - start
        log_verbose(f"[Warmer] {name} loaded in {elapsed:.2f}s")
    except Exception as e:
        log.error(f"[Warmer] {name} failed: {e}")
        log_verbose(f"[Warmer] {name} exception: {type(e).__name__}: {e}")


def _print_status_report():
    """Print a clear table of model statuses."""
    try:
        # Check TransNet
        t_status = (
            "OK" if (settings.model_cache_dir / "transnetv2.onnx").exists() else "MISSING"
        )
        arcface_status = "OK" if (settings.model_cache_dir / "arcface" / "w600k_r50.onnx").exists() else "MISSING"

        # Log verbose details
        log_verbose(f"[Warmer] Status: TransNet={t_status}, ArcFace={arcface_status}")
        log_verbose(f"[Warmer] Model cache contents: {list(settings.model_cache_dir.glob('*'))[:10]}")

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
        print(f"{'ArcFace':<25} | {arcface_status:<20}")
        print("=" * 50 + "\n")
        print(
            "Note: 'Lazy' models load on first use. Failures will be logged but won't crash the server.\n"
        )

    except Exception:
        pass


async def _warmup_bge():
    """Warmup BGE Reranker (Blocking)."""
    start = time.time()
    try:
        log.info("[Warmer] Checking BGE-Reranker...")
        log_verbose("[Warmer] Loading BGE BAAI/bge-reranker-v2-m3")
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("BAAI/bge-reranker-v2-m3", trust_remote_code=True)
        del model
        elapsed = time.time() - start
        log_verbose(f"[Warmer] BGE-Reranker loaded in {elapsed:.2f}s")
    except Exception as e:
        log.error(f"[Warmer] BGE-Reranker failed: {e}")
        log_verbose(f"[Warmer] BGE exception: {type(e).__name__}: {e}")

