import asyncio
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.utils.resource_arbiter import GPU_SEMAPHORE

log = logging.getLogger(__name__)


class VisualEncoderInterface(ABC):
    """Abstract base class for visual encoders."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def encode_image(
        self, image: np.ndarray | bytes | Path
    ) -> list[float]: ...

    @abstractmethod
    async def encode_batch(
        self, images: list[np.ndarray | bytes | Path]
    ) -> list[list[float]]: ...

    def cleanup(self) -> None: ...


class CLIPEncoder(VisualEncoderInterface):
    """OpenAI CLIP visual encoder with async support."""

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        self._model_name = model_name
        self._pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._dim = 768 if "L" in model_name else 512
        self._init_lock = asyncio.Lock()

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"CLIP-{self._model_name}"

    async def _lazy_init(self):
        if self._model is not None:
            return
        async with self._init_lock:
            if self._model is not None:
                return

            def _load():
                log.info(f"[VisualEncoder] Initializing {self.name}...")
                try:
                    import open_clip

                    model, _, preprocess = open_clip.create_model_and_transforms(
                        self._model_name, pretrained=self._pretrained
                    )
                    model.eval()
                    import torch

                    if torch.cuda.is_available():
                        model = model.cuda()
                    return model, preprocess
                except ImportError:
                    from transformers import CLIPModel, CLIPProcessor

                    hf_model = "openai/clip-vit-large-patch14"
                    preprocess = CLIPProcessor.from_pretrained(hf_model)
                    model = CLIPModel.from_pretrained(hf_model)
                    model.eval()
                    import torch

                    if torch.cuda.is_available():
                        model = model.cuda()
                    return model, preprocess

            self._model, self._preprocess = await asyncio.to_thread(_load)

    async def encode_image(
        self, image: np.ndarray | bytes | Path
    ) -> list[float]:
        await self._lazy_init()

        def _process():
            # Load and preprocess image in thread
            if isinstance(image, Path):
                pil_img = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                pil_img = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                pil_img = Image.fromarray(image).convert("RGB")


            if hasattr(
                self._preprocess, "feature_extractor"
            ) or hasattr(self._preprocess, "image_processor"):
                inputs = self._preprocess(images=pil_img, return_tensors="pt")
            else:
                inputs = self._preprocess(pil_img).unsqueeze(0)
            return inputs

        inputs = await asyncio.to_thread(_process)

        async with GPU_SEMAPHORE:
            def _infer():
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                if isinstance(inputs, dict):
                    # Transformers
                    gpu_inputs = {
                        k: v.to(device)
                        for k, v in inputs.items()
                        if hasattr(v, "to")
                    }
                    with torch.no_grad():
                        feats = self._model.get_image_features(**gpu_inputs)
                else:
                    # OpenCLIP
                    gpu_inputs = inputs.to(device)
                    with torch.no_grad():
                        feats = self._model.encode_image(gpu_inputs)

                feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats.cpu().numpy().flatten().tolist()

            return await asyncio.to_thread(_infer)

    async def encode_batch(
        self, images: list[np.ndarray | bytes | Path]
    ) -> list[list[float]]:
        await self._lazy_init()

        async def _load_and_preprocess(img):
            def __task():
                if isinstance(img, Path):
                    pil = Image.open(img).convert("RGB")
                elif isinstance(img, bytes):
                    pil = Image.open(io.BytesIO(img)).convert("RGB")
                else:
                    pil = Image.fromarray(img).convert("RGB")

                if hasattr(
                    self._preprocess, "feature_extractor"
                ) or hasattr(self._preprocess, "image_processor"):
                    return pil  # Keep as PIL for transformers batch processing
                else:
                    return self._preprocess(pil)  # Tensor for OpenCLIP

            return await asyncio.to_thread(__task)

        # 1. Prepare inputs in parallel
        preprocessed_items = await asyncio.gather(
            *[_load_and_preprocess(img) for img in images]
        )

        async with GPU_SEMAPHORE:
            def _infer_batch():
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                if hasattr(
                    self._preprocess, "feature_extractor"
                ) or hasattr(self._preprocess, "image_processor"):
                    # Transformers
                    inputs = self._preprocess(
                        images=preprocessed_items, return_tensors="pt"
                    )
                    inputs = {
                        k: v.to(device)
                        for k, v in inputs.items()
                        if hasattr(v, "to")
                    }
                    with torch.no_grad():
                        feats = self._model.get_image_features(**inputs)
                else:
                    # OpenCLIP
                    batch = torch.stack(preprocessed_items).to(device)
                    with torch.no_grad():
                        feats = self._model.encode_image(batch)

                feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats.cpu().numpy().tolist()

            return await asyncio.to_thread(_infer_batch)

    def cleanup(self) -> None:
        if self._model is not None:
            import gc
            import torch

            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class SigLIPEncoder(VisualEncoderInterface):
    """Google SigLIP visual encoder (SOTA) with async support."""

    def __init__(self, model_name: str = "ViT-SO400M-14-SigLIP-384"):
        self._model_name = model_name
        self._model = None
        self._preprocess = None
        self._dim = 1152
        self._init_lock = asyncio.Lock()

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"SigLIP-{self._model_name}"

    async def _lazy_init(self):
        if self._model is not None:
            return
        async with self._init_lock:
            if self._model is not None:
                return

            def _load():
                log.info(f"[VisualEncoder] Initializing {self.name}...")
                try:
                    import open_clip
                except ImportError:
                    log.error("[VisualEncoder] open_clip not installed. Install with `pip install open_clip_torch`. Visual features will be disabled.")
                    return None, None

                # Try OpenCLIP first
                try:
                    import open_clip
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        self._model_name, pretrained="webli"
                    )
                    model.eval()
                    import torch
                    if torch.cuda.is_available():
                        model = model.cuda()
                    return model, preprocess
                except Exception as e:
                    log.warning(f"[VisualEncoder] OpenCLIP load failed ({e}). Trying Transformers fallback...")

                # Transformers fallback for SigLIP
                try:
                    from transformers import AutoModel, AutoProcessor
                    hf_model = "google/siglip-so400m-patch14-384"
                    model = AutoModel.from_pretrained(hf_model)
                    preprocess = AutoProcessor.from_pretrained(hf_model)
                    model.eval()
                    import torch
                    if torch.cuda.is_available():
                        model = model.cuda()
                    return model, preprocess
                except Exception as e2:
                    log.error(f"[VisualEncoder] SigLIP Fallback failed: {e2}")
                    return None, None

            self._model, self._preprocess = await asyncio.to_thread(_load)

    async def encode_image(
        self, image: np.ndarray | bytes | Path
    ) -> list[float]:
        await self._lazy_init()
        if self._model is None:
            return []  # Return empty if load failed

        def _process():
            if isinstance(image, Path):
                pil_img = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                pil_img = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                pil_img = Image.fromarray(image).convert("RGB")
            return self._preprocess(pil_img).unsqueeze(0)

        inputs = await asyncio.to_thread(_process)

        async with GPU_SEMAPHORE:
            def _infer():
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                gpu_inputs = inputs.to(device)
                with torch.no_grad():
                    feats = self._model.encode_image(gpu_inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats.cpu().numpy().flatten().tolist()

            return await asyncio.to_thread(_infer)

    async def encode_batch(
        self, images: list[np.ndarray | bytes | Path]
    ) -> list[list[float]]:
        await self._lazy_init()

        async def _prep(img):
            def __task():
                if isinstance(img, Path):
                    pil = Image.open(img).convert("RGB")
                elif isinstance(img, bytes):
                    pil = Image.open(io.BytesIO(img)).convert("RGB")
                else:
                    pil = Image.fromarray(img).convert("RGB")
                return self._preprocess(pil)

            return await asyncio.to_thread(__task)

        tensors = await asyncio.gather(*[_prep(img) for img in images])

        async with GPU_SEMAPHORE:
            def _infer_batch():
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                batch = torch.stack(tensors).to(device)
                with torch.no_grad():
                    feats = self._model.encode_image(batch)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats.cpu().numpy().tolist()

            return await asyncio.to_thread(_infer_batch)

    def cleanup(self) -> None:
        if self._model is not None:
            import gc
            import torch

            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class VLMVisionTowerEncoder(VisualEncoderInterface):
    """Vision features from VLM, reusing the loaded model."""

    def __init__(self, vlm_client: Any):
        self._vlm = vlm_client
        self._dim = getattr(vlm_client, "vision_dim", 1024)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "VLM-VisionTower"

    async def encode_image(self, image: np.ndarray | bytes | Path) -> list[float]:
        # Implementation depends on VLM client
        if hasattr(self._vlm, "encode_image"):
            return await self._vlm.encode_image(image)
        elif hasattr(self._vlm, "get_visual_features"):
            features = await self._vlm.get_visual_features(image)
            return (
                features.tolist() if hasattr(features, "tolist") else list(features)
            )
        else:
            raise NotImplementedError("VLM client doesn't support vision features")

    async def encode_batch(
        self, images: list[np.ndarray | bytes | Path]
    ) -> list[list[float]]:
        return await asyncio.gather(*[self.encode_image(img) for img in images])


def get_default_visual_encoder() -> VisualEncoderInterface:
    """Get the default visual encoder (singleton)."""
    global _visual_encoder
    if _visual_encoder is not None:
        return _visual_encoder

    from config import settings

    encoder_type = getattr(settings, "visual_encoder_type", "siglip")
    if encoder_type == "siglip":
        _visual_encoder = SigLIPEncoder()
    else:
        _visual_encoder = CLIPEncoder()

    return _visual_encoder


_visual_encoder: VisualEncoderInterface | None = None
