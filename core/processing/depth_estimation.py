from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class DepthEstimator:
    def __init__(self, model_name: str = "depth-anything-v2-small"):
        self.model_name = model_name
        self.model = None
        self._device = None
        self._init_lock = asyncio.Lock()

    def _get_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    async def _lazy_load(self) -> bool:
        if self.model is not None:
            return True

        async with self._init_lock:
            if self.model is not None:
                return True
            try:
                from core.utils.resource_arbiter import RESOURCE_ARBITER

                async with RESOURCE_ARBITER.acquire("depth", vram_gb=1.0):  # Reduced from 1.5 with fp16
                    log.info(f"[DepthEstimator] Loading {self.model_name}...")
                    import torch
                    from transformers import pipeline

                    device = self._get_device()
                    # Load in fp16 to reduce VRAM by ~50%
                    self.model = pipeline(
                        "depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device=0 if device == "cuda" else -1,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    )
                    self._device = device
                    log.info(f"[DepthEstimator] Loaded on {device}")
                    return True
            except ImportError as e:
                log.warning(f"[DepthEstimator] Dependencies missing: {e}")
                return False
            except Exception as e:
                log.error(f"[DepthEstimator] Load failed: {e}")
                return False

    async def estimate_depth(self, image: np.ndarray | Path) -> dict[str, Any]:
        if not await self._lazy_load():
            return {"depth_map": None, "error": "Model not loaded"}

        try:
            from PIL import Image

            if isinstance(image, Path):
                img = Image.open(image)
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                img = image

            result = self.model(img)
            depth_map = np.array(result["depth"])

            stats = {
                "min_depth": float(np.min(depth_map)),
                "max_depth": float(np.max(depth_map)),
                "mean_depth": float(np.mean(depth_map)),
                "std_depth": float(np.std(depth_map)),
            }

            log.info(f"[DepthEstimator] Range: {stats['min_depth']:.2f} - {stats['max_depth']:.2f}")
            return {"depth_map": depth_map, "stats": stats}

        except Exception as e:
            log.error(f"[DepthEstimator] Estimation failed: {e}")
            return {"depth_map": None, "error": str(e)}

    async def estimate_object_distance(
        self,
        image: np.ndarray | Path,
        bbox: tuple[int, int, int, int],
        focal_length_px: float = 1000.0,
        real_height_cm: float = 170.0,
    ) -> dict[str, Any]:
        result = await self.estimate_depth(image)
        if result.get("error"):
            return result

        depth_map = result["depth_map"]
        x1, y1, x2, y2 = bbox

        h, w = depth_map.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        region_depth = depth_map[y1:y2, x1:x2]
        if region_depth.size == 0:
            return {"distance_cm": None, "error": "Invalid bbox"}

        median_depth = float(np.median(region_depth))
        estimated_distance_cm = (focal_length_px * real_height_cm) / ((y2 - y1) + 1)

        return {
            "relative_depth": median_depth,
            "distance_cm": estimated_distance_cm,
            "distance_m": estimated_distance_cm / 100,
            "confidence": 0.7,
        }

    async def matches_distance_constraint(
        self,
        image: np.ndarray | Path,
        bbox: tuple[int, int, int, int],
        min_distance_m: float | None = None,
        max_distance_m: float | None = None,
        semantic_distance: str | None = None,
        focal_length_px: float = 1000.0,
        real_height_cm: float = 170.0,
    ) -> dict[str, Any]:
        result = await self.estimate_object_distance(
            image, bbox, focal_length_px, real_height_cm
        )
        if result.get("error"):
            return result

        distance_m = result["distance_m"]

        if semantic_distance:
            semantic_lower = semantic_distance.lower()
            if semantic_lower in ("very close", "touching", "adjacent"):
                min_distance_m = min_distance_m or 0.0
                max_distance_m = max_distance_m or 0.5
            elif semantic_lower in ("close", "near", "nearby"):
                min_distance_m = min_distance_m or 0.5
                max_distance_m = max_distance_m or 2.0
            elif semantic_lower in ("medium", "moderate"):
                min_distance_m = min_distance_m or 2.0
                max_distance_m = max_distance_m or 5.0
            elif semantic_lower in ("far", "distant"):
                min_distance_m = min_distance_m or 5.0
                max_distance_m = max_distance_m or 20.0
            elif semantic_lower in ("very far", "background"):
                min_distance_m = min_distance_m or 10.0
                max_distance_m = None

        matches = True
        if min_distance_m is not None and distance_m < min_distance_m:
            matches = False
        if max_distance_m is not None and distance_m > max_distance_m:
            matches = False

        return {
            "matches": matches,
            "distance_m": distance_m,
            "min_constraint": min_distance_m,
            "max_constraint": max_distance_m,
            "semantic_distance": semantic_distance,
        }

    def cleanup(self) -> None:
        if self.model:
            del self.model
            self.model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("[DepthEstimator] Resources released")
