from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class ClothingAttributeDetector:
    def __init__(self):
        self.model = None
        self.processor = None
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

                async with RESOURCE_ARBITER.acquire("clothing_clip", vram_gb=1.5):
                    log.info("[ClothingDetector] Loading CLIP for open-vocabulary detection...")
                    import torch
                    from transformers import CLIPModel, CLIPProcessor

                    device = self._get_device()
                    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                    self.model = self.model.to(device).eval()
                    self._device = device
                    log.info(f"[ClothingDetector] CLIP loaded on {device}")
                    return True

            except ImportError as e:
                log.warning(f"[ClothingDetector] Dependencies missing: {e}")
                return False
            except Exception as e:
                log.error(f"[ClothingDetector] Load failed: {e}")
                return False

    async def match_description(
        self,
        image: np.ndarray | Path,
        target_description: str,
    ) -> dict[str, Any]:
        if not await self._lazy_load():
            return {"score": 0.0, "error": "Model not loaded"}

        try:
            import torch
            from PIL import Image

            if isinstance(image, Path):
                img = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert("RGB")
            else:
                img = image

            prompts = [
                f"a person wearing {target_description}",
                f"a photo of {target_description}",
                target_description,
            ]

            inputs = self.processor(
                text=prompts, images=img, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()

            best_score = float(max(probs))
            log.info(f"[ClothingDetector] '{target_description}' score: {best_score:.3f}")

            return {
                "target": target_description,
                "score": best_score,
                "match": best_score > 0.25,
            }

        except Exception as e:
            log.error(f"[ClothingDetector] Detection failed: {e}")
            return {"score": 0.0, "error": str(e)}

    async def match_constraints(
        self,
        image: np.ndarray | Path,
        color: str | None = None,
        item: str | None = None,
        pattern: str | None = None,
        side: str | None = None,
    ) -> dict[str, Any]:
        parts = []
        if color:
            parts.append(color)
        if pattern and pattern != "solid":
            parts.append(pattern)
        if item:
            parts.append(item)
        if side and side not in ("unknown", "both"):
            parts.append(f"on {side} side")

        if not parts:
            return {"score": 0.0, "error": "No constraints provided"}

        description = " ".join(parts)
        return await self.match_description(image, description)

    async def compare_candidates(
        self,
        image: np.ndarray | Path,
        candidate_descriptions: list[str],
    ) -> list[dict[str, Any]]:
        if not await self._lazy_load():
            return [{"label": c, "score": 0.0} for c in candidate_descriptions]

        try:
            import torch
            from PIL import Image

            if isinstance(image, Path):
                img = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert("RGB")
            else:
                img = image

            prompts = [f"a person wearing {c}" for c in candidate_descriptions]

            inputs = self.processor(
                text=prompts, images=img, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()

            results = [
                {"label": candidate_descriptions[i], "score": float(probs[i])}
                for i in range(len(candidate_descriptions))
            ]
            results.sort(key=lambda x: x["score"], reverse=True)

            return results

        except Exception as e:
            log.error(f"[ClothingDetector] Comparison failed: {e}")
            return [{"label": c, "score": 0.0, "error": str(e)} for c in candidate_descriptions]

    def cleanup(self) -> None:
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("[ClothingDetector] Resources released")
