"""Speed estimation using optical flow (RAFT)."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class SpeedEstimator:
    """Estimates speed of objects or regions using optical flow."""

    def __init__(self, model_name: str = "raft-small"):
        """Initialize speed estimator with RAFT model."""
        self.model_name = model_name
        self.model = None
        self._device = None
        self._init_lock = asyncio.Lock()
        self._transforms = None

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

                async with RESOURCE_ARBITER.acquire(
                    "raft", vram_gb=0.5
                ):  # Reduced with fp16
                    log.info("[SpeedEstimator] Loading RAFT optical flow...")
                    import torchvision.models.optical_flow as of

                    device = self._get_device()
                    if self.model_name == "raft-large":
                        weights = of.Raft_Large_Weights.DEFAULT
                        self.model = of.raft_large(weights=weights)
                    else:
                        weights = of.Raft_Small_Weights.DEFAULT
                        self.model = of.raft_small(weights=weights)

                    self.model = self.model.to(device)
                    if device == "cuda":
                        self.model = self.model.half()  # fp16 to reduce VRAM
                    self.model.eval()
                    self._device = device
                    self._transforms = weights.transforms()
                    log.info(
                        f"[SpeedEstimator] Loaded {self.model_name} on {device}"
                    )
                    return True

            except ImportError as e:
                log.warning(f"[SpeedEstimator] Dependencies missing: {e}")
                return False
            except Exception as e:
                log.error(f"[SpeedEstimator] Load failed: {e}")
                return False

    async def compute_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> dict[str, Any]:
        """Compute optical flow between two frames."""
        if not await self._lazy_load():
            return {"flow": None, "error": "Model not loaded"}

        try:
            import torch

            if frame1.shape != frame2.shape:
                return {"flow": None, "error": "Frame shapes must match"}

            if len(frame1.shape) == 3 and frame1.shape[2] == 3:
                f1 = (
                    torch.from_numpy(frame1)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                )
                f2 = (
                    torch.from_numpy(frame2)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                )
            else:
                return {"flow": None, "error": "Frames must be RGB"}

            if self._transforms is None:
                return {"flow": None, "error": "Transforms not loaded"}

            f1, f2 = self._transforms(f1, f2)
            f1 = f1.to(self._device)
            f2 = f2.to(self._device)

            # Convert to fp16 if model is in fp16
            if self._device == "cuda":
                f1 = f1.half()
                f2 = f2.half()

            with torch.no_grad():
                if self.model is None:
                    return {"flow": None, "error": "Model lost"}

                flow_predictions = self.model(f1, f2)
                flow = flow_predictions[-1][0].cpu().numpy()

            flow_magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2)

            return {
                "flow": flow,
                "magnitude": flow_magnitude,
                "mean_velocity_px": float(np.mean(flow_magnitude)),
                "max_velocity_px": float(np.max(flow_magnitude)),
            }

        except Exception as e:
            log.error(f"[SpeedEstimator] Flow computation failed: {e}")
            return {"flow": None, "error": str(e)}

    async def estimate_speed(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        fps: float = 30.0,
        scale_px_per_meter: float = 100.0,
    ) -> dict[str, Any]:
        """Estimate real-world speed from optical flow."""
        result = await self.compute_optical_flow(frame1, frame2)
        if result.get("error"):
            return result

        mean_px = result["mean_velocity_px"]
        max_px = result["max_velocity_px"]

        time_interval = 1.0 / fps
        mean_speed_m_s = (mean_px / scale_px_per_meter) / time_interval
        max_speed_m_s = (max_px / scale_px_per_meter) / time_interval

        return {
            "mean_speed_m_s": mean_speed_m_s,
            "max_speed_m_s": max_speed_m_s,
            "mean_speed_km_h": mean_speed_m_s * 3.6,
            "max_speed_km_h": max_speed_m_s * 3.6,
            "mean_velocity_px": mean_px,
            "max_velocity_px": max_px,
        }

    async def matches_speed_constraint(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        min_speed_km_h: float | None = None,
        max_speed_km_h: float | None = None,
        semantic_speed: str | None = None,
        fps: float = 30.0,
        scale_px_per_meter: float = 100.0,
    ) -> dict[str, Any]:
        """Check if motion matches speed constraints."""
        result = await self.estimate_speed(
            frame1, frame2, fps, scale_px_per_meter
        )
        if result.get("error"):
            return result

        speed = result["mean_speed_km_h"]

        if semantic_speed:
            semantic_lower = semantic_speed.lower()
            if semantic_lower in ("stationary", "still", "stopped"):
                min_speed_km_h = min_speed_km_h or 0.0
                max_speed_km_h = max_speed_km_h or 0.5
            elif semantic_lower in ("slow", "walking", "crawling"):
                min_speed_km_h = min_speed_km_h or 0.5
                max_speed_km_h = max_speed_km_h or 5.0
            elif semantic_lower in ("normal", "jogging", "moderate"):
                min_speed_km_h = min_speed_km_h or 5.0
                max_speed_km_h = max_speed_km_h or 15.0
            elif semantic_lower in ("fast", "running", "quick"):
                min_speed_km_h = min_speed_km_h or 15.0
                max_speed_km_h = max_speed_km_h or 50.0
            elif semantic_lower in ("very fast", "sprinting", "racing"):
                min_speed_km_h = min_speed_km_h or 30.0
                max_speed_km_h = None

        matches = True
        if min_speed_km_h is not None and speed < min_speed_km_h:
            matches = False
        if max_speed_km_h is not None and speed > max_speed_km_h:
            matches = False

        return {
            "matches": matches,
            "speed_km_h": speed,
            "min_constraint": min_speed_km_h,
            "max_constraint": max_speed_km_h,
            "semantic_speed": semantic_speed,
        }

    async def estimate_region_speed(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        bbox: tuple[int, int, int, int],
        fps: float = 30.0,
        scale_px_per_meter: float = 100.0,
    ) -> dict[str, Any]:
        """Estimate speed within a specific bounding box."""
        result = await self.compute_optical_flow(frame1, frame2)
        if result.get("error"):
            return result

        x1, y1, x2, y2 = bbox
        magnitude = result["magnitude"]
        h, w = magnitude.shape

        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        region = magnitude[y1:y2, x1:x2]
        if region.size == 0:
            return {"error": "Invalid bbox"}

        mean_px = float(np.mean(region))
        max_px = float(np.max(region))

        time_interval = 1.0 / fps
        mean_speed_m_s = (mean_px / scale_px_per_meter) / time_interval

        return {
            "region_mean_speed_m_s": mean_speed_m_s,
            "region_mean_speed_km_h": mean_speed_m_s * 3.6,
            "region_max_velocity_px": max_px,
            "bbox": bbox,
        }

    def cleanup(self) -> None:
        """Release resources."""
        if self.model:
            del self.model
            self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("[SpeedEstimator] Resources released")
