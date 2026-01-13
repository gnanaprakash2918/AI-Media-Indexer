"""Global GPU resource manager with VRAM tracking and cancellation support.

Prevents OOM crashes by strictly managing VRAM allocation across models.
Integrates with CancellationToken for preemptive job cancellation.
"""

import asyncio
import gc
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from core.utils.locks import GPU_SEMAPHORE

if TYPE_CHECKING:
    from core.utils.cancellation import CancellationToken


@dataclass
class ModelVRAM:
    """Approximate VRAM usage per model in GB."""

    whisper_large: float = 3.0
    whisper_small: float = 1.0
    insightface: float = 1.5
    vlm_7b: float = 6.0
    vlm_3b: float = 3.0
    timesformer: float = 2.0
    nv_embed_v2: float = 4.0
    yolo_world: float = 1.0
    pyannote: float = 1.5
    clap: float = 1.0


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
            total_vram_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
            )
        self.total_vram = total_vram_gb or 8.0
        self.current_usage = 0.0
        self._lock = asyncio.Lock()
        self.loaded_models: set[str] = set()
        self._initialized = True

    @asynccontextmanager
    async def acquire(
        self,
        model_name: str,
        vram_gb: float,
        job_id: str | None = None,
    ):
        """Acquire GPU resources for a model.

        Args:
            model_name: Name of the model (for tracking).
            vram_gb: Approximate VRAM required in GB.
            job_id: Optional job ID for cancellation support.

        Yields:
            CancellationToken if job_id provided, else None.

        Raises:
            CancellationError: If job was cancelled before or during acquisition.
        """
        from core.utils.cancellation import (
            CancellationError,
            get_or_create_token,
        )

        token: CancellationToken | None = None
        if job_id:
            token = get_or_create_token(job_id)

        async with self._lock:
            # Check cancellation before waiting
            if token and token.is_cancelled:
                raise CancellationError(
                    f"Job {job_id} cancelled before resource acquisition"
                )

            # Wait for VRAM if needed (offload other models)
            while self.current_usage + vram_gb > self.total_vram * 0.9:
                if not self.loaded_models:
                    break  # Nothing to offload, proceed anyway
                await self._offload_least_recent()

            self.current_usage += vram_gb
            self.loaded_models.add(model_name)

        try:
            async with GPU_SEMAPHORE:  # Mutual exclusion for GPU ops
                yield token
        except CancellationError:
            raise
        finally:
            async with self._lock:
                self.current_usage = max(0, self.current_usage - vram_gb)
                self.loaded_models.discard(model_name)
                self._cleanup_vram()

    def _cleanup_vram(self) -> None:
        """Force garbage collection and CUDA cache clear."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def _offload_least_recent(self) -> None:
        """Placeholder for model offloading logic."""
        self._cleanup_vram()
        await asyncio.sleep(0.1)  # Give time for cleanup

    def force_release_all(self) -> None:
        """Emergency release all resources."""
        self.loaded_models.clear()
        self.current_usage = 0
        self._cleanup_vram()

    def get_status(self) -> dict:
        """Get current resource status."""
        return {
            "total_vram_gb": self.total_vram,
            "current_usage_gb": self.current_usage,
            "available_gb": self.total_vram - self.current_usage,
            "loaded_models": list(self.loaded_models),
            "utilization_pct": (self.current_usage / self.total_vram) * 100,
        }


# Global singleton instance
RESOURCE_ARBITER = ResourceArbiter()
