"""Global GPU resource manager with VRAM tracking and cancellation support.

Prevents OOM crashes by strictly managing VRAM allocation across models.
Integrates with CancellationToken for preemptive job cancellation.
"""

import asyncio
import gc
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    pass


@dataclass
class ModelVRAM:
    """Approximate VRAM usage per model in GB (fp16 where applicable)."""

    whisper_large: float = 3.0
    whisper_small: float = 1.0
    insightface: float = 1.5
    vlm_7b: float = 6.0
    vlm_3b: float = 3.0
    timesformer: float = 1.0  # fp16
    nv_embed_v2: float = 4.0
    yolo_world: float = 1.0
    pyannote: float = 1.5
    clap: float = 1.0
    depth_anything: float = 1.0  # fp16
    raft: float = 0.5  # fp16


class ResourceArbiter:
    """Global GPU resource manager with VRAM tracking.

    Usage:
        async with RESOURCE_ARBITER.acquire("vlm", vram_gb=6.0, job_id="job_123"):
            result = await vlm.predict(frame)
        # Model automatically released, VRAM cleaned
    """

    def __init__(self, total_vram_gb: float | None = None):
        """Initialize ResourceArbiter.

        Args:
            total_vram_gb: Total VRAM available. Auto-detected if None.
        """
        if total_vram_gb is None and torch.cuda.is_available():
            try:
                total_vram_gb = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
            except Exception:
                total_vram_gb = 8.0

        self.total_vram = total_vram_gb or 8.0
        self.current_usage = 0.0
        self._lock = asyncio.Lock()

        # Track loaded models and their unload callbacks
        # Format: {model_name: {"vram": float, "unload_fn": callable, "last_used": float}}
        self.registry: dict[str, dict] = {}
        self._gpu_semaphore = asyncio.Semaphore(1)
        self._initialized = True

    def register_model(self, model_name: str, unload_fn: Callable) -> None:
        """Register a model's unload function for VRAM management."""
        if model_name not in self.registry:
            self.registry[model_name] = {
                "vram": 0.0,
                "unload_fn": unload_fn,
                "last_used": 0.0,
                "active": False,
            }

    @asynccontextmanager
    async def acquire(
        self,
        model_name: str,
        vram_gb: float,
        job_id: str | None = None,
    ):
        """Acquire GPU resources for a model.

        Args:
            model_name: Name of the model.
            vram_gb: Approximate VRAM required in GB.
            job_id: Optional job ID for cancellation support.
        """
        import time

        from core.utils.cancellation import (
            CancellationError,
            get_or_create_token,
        )

        import logging

        logger = logging.getLogger(__name__)

        token = get_or_create_token(job_id) if job_id else None

        logger.info(f"[Arbiter] Requesting {model_name} ({vram_gb}GB)...")
        async with self._lock:
            # Check cancellation
            if token and token.is_cancelled:
                raise CancellationError(f"Job {job_id} cancelled")

            # Update registry
            if model_name not in self.registry:
                # If not registered with callback, just track basic usage
                self.registry[model_name] = {
                    "vram": vram_gb,
                    "unload_fn": None,
                    "last_used": time.time(),
                    "active": False,
                }

            # Wait for VRAM availability (simple greedy approach)
            limit = self.total_vram * 0.9

            while self.current_usage + vram_gb > limit:
                logger.info(
                    f"[Arbiter] VRAM full ({self.current_usage}/{limit}), offloading..."
                )
                # Try to offload least recent INACTIVE model
                offloaded = await self._offload_least_recent()
                if not offloaded:
                    logger.warning(
                        "[Arbiter] Could not offload any models, proceeding anyway"
                    )
                    break
                logger.info("[Arbiter] Offload successful")

            self.current_usage += vram_gb
            self.registry[model_name]["active"] = True
            self.registry[model_name]["vram"] = vram_gb
            self.registry[model_name]["last_used"] = time.time()

        try:
            logger.info(
                f"[Arbiter] Waiting for GPU semaphore ({model_name})..."
            )
            async with self._gpu_semaphore:
                logger.info(f"[Arbiter] GPU semaphore acquired ({model_name})")
                yield token
                logger.info(f"[Arbiter] Releasing GPU semaphore ({model_name})")
        except CancellationError:
            raise
        finally:
            logger.info(f"[Arbiter] Cleaning up {model_name}...")
            async with self._lock:
                # Mark as inactive AND release VRAM allocation
                self.registry[model_name]["active"] = False
                self.registry[model_name]["last_used"] = time.time()
                # CRITICAL FIX: Decrement current_usage to free VRAM budget
                model_vram = self.registry[model_name].get("vram", vram_gb)
                self.current_usage = max(0, self.current_usage - model_vram)
                logger.info(
                    f"[Arbiter] Released {model_vram}GB VRAM, current usage: {self.current_usage:.1f}GB"
                )
            logger.info(f"[Arbiter] Cleanup complete {model_name}")

    async def _offload_least_recent(self) -> bool:
        """Finds and unloads the least recently used inactive model."""
        candidates = [
            (name, data)
            for name, data in self.registry.items()
            if not data["active"] and data["unload_fn"]
        ]

        if not candidates:
            return False

        # Sort by last_used (oldest first)
        candidates.sort(key=lambda x: x[1]["last_used"])

        name, data = candidates[0]
        unload_fn = data["unload_fn"]
        vram = data["vram"]

        try:
            # Call the unload callback
            if asyncio.iscoroutinefunction(unload_fn):
                await unload_fn()
            else:
                unload_fn()

            self.current_usage = max(0, self.current_usage - vram)
            # Remove from registry or check "loaded" flag?
            # Ideally we keep it in registry but mark as unloaded.
            # For now, simplistic approach: leave in registry, but usage is what matters.
            # We assume unload_fn clears the memory.

            self._cleanup_vram()
            return True
        except Exception:
            return False

    def _cleanup_vram(self) -> None:
        """Force garbage collection and CUDA cache clear."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def force_release_all(self) -> None:
        """Emergency release all resources."""
        self.current_usage = 0
        self._cleanup_vram()

    def get_status(self) -> dict:
        """Get current resource status."""
        return {
            "total_vram_gb": self.total_vram,
            "current_usage_gb": self.current_usage,
            "available_gb": self.total_vram - self.current_usage,
            "models": {
                k: {"active": v["active"], "last": v["last_used"]}
                for k, v in self.registry.items()
            },
        }


# Global singleton instance
RESOURCE_ARBITER = ResourceArbiter()
GPU_SEMAPHORE = RESOURCE_ARBITER._gpu_semaphore  # Backward compatibility
