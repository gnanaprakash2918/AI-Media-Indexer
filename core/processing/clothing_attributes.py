from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)

CLOTHING_COLORS = [
    "black", "white", "gray", "red", "blue", "green", "yellow", "orange",
    "pink", "purple", "brown", "beige", "navy", "maroon", "teal", "gold",
]

CLOTHING_ITEMS = [
    "t-shirt", "shirt", "jacket", "coat", "sweater", "hoodie", "blazer",
    "pants", "jeans", "shorts", "skirt", "dress", "suit",
    "shoes", "sneakers", "boots", "sandals", "heels",
    "hat", "cap", "scarf", "tie", "watch", "glasses", "bag",
]

CLOTHING_PATTERNS = [
    "solid", "striped", "checkered", "plaid", "polka-dot", "floral",
    "camouflage", "denim", "leather", "printed",
]


class ClothingAttributeDetector:
    def __init__(self, model_name: str = "clip"):
        self.model_name = model_name
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

                async with RESOURCE_ARBITER.acquire("clothing", vram_gb=1.5):
                    log.info("[ClothingDetector] Loading CLIP for zero-shot clothing...")
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

    async def detect_clothing(
        self,
        image: np.ndarray | Path,
        body_region: Literal["full", "upper", "lower", "feet"] = "full",
        top_k: int = 3,
    ) -> dict[str, Any]:
        if not await self._lazy_load():
            return {"attributes": [], "error": "Model not loaded"}

        try:
            import torch
            from PIL import Image

            if isinstance(image, Path):
                img = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert("RGB")
            else:
                img = image

            color_prompts = [f"a person wearing {c} clothing" for c in CLOTHING_COLORS]
            item_prompts = [f"a person wearing a {i}" for i in CLOTHING_ITEMS]
            pattern_prompts = [f"clothing with {p} pattern" for p in CLOTHING_PATTERNS]

            results = {
                "colors": await self._classify(img, color_prompts, CLOTHING_COLORS, top_k),
                "items": await self._classify(img, item_prompts, CLOTHING_ITEMS, top_k),
                "patterns": await self._classify(img, pattern_prompts, CLOTHING_PATTERNS, top_k),
                "body_region": body_region,
            }

            top_color = results["colors"][0]["label"] if results["colors"] else "unknown"
            top_item = results["items"][0]["label"] if results["items"] else "unknown"
            results["description"] = f"{top_color} {top_item}"

            log.info(f"[ClothingDetector] Detected: {results['description']}")
            return results

        except Exception as e:
            log.error(f"[ClothingDetector] Detection failed: {e}")
            return {"attributes": [], "error": str(e)}

    async def _classify(
        self,
        image,
        prompts: list[str],
        labels: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        import torch

        inputs = self.processor(
            text=prompts, images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        top_indices = probs.argsort()[::-1][:top_k]
        return [
            {"label": labels[i], "confidence": float(probs[i])}
            for i in top_indices
        ]

    async def match_clothing_query(
        self,
        image: np.ndarray | Path,
        color: str | None = None,
        item: str | None = None,
        pattern: str | None = None,
    ) -> dict[str, Any]:
        detection = await self.detect_clothing(image)
        if detection.get("error"):
            return detection

        match_score = 0.0
        matches = []

        if color:
            color_lower = color.lower()
            for c in detection["colors"]:
                if c["label"] == color_lower:
                    match_score += c["confidence"]
                    matches.append(f"color:{color_lower}")
                    break

        if item:
            item_lower = item.lower()
            for i in detection["items"]:
                if i["label"] == item_lower or item_lower in i["label"]:
                    match_score += i["confidence"]
                    matches.append(f"item:{item_lower}")
                    break

        if pattern:
            pattern_lower = pattern.lower()
            for p in detection["patterns"]:
                if p["label"] == pattern_lower:
                    match_score += p["confidence"]
                    matches.append(f"pattern:{pattern_lower}")
                    break

        constraints_count = sum([1 for x in [color, item, pattern] if x])
        normalized_score = match_score / constraints_count if constraints_count > 0 else 0

        return {
            "match_score": normalized_score,
            "matches": matches,
            "detection": detection,
        }

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
