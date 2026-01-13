"""SOTA Visual Embeddings using SigLIP.

SigLIP (Sigmoid Loss for Language-Image Pre-training) outperforms CLIP
for image-text matching tasks. This module provides visual embeddings
for cross-modal search (text query → image results).

Key advantages over CLIP:
- Better handling of hard negatives
- Superior zero-shot retrieval performance
- More robust to noise
"""

import gc
from pathlib import Path

import torch
from PIL import Image

from config import settings
from core.utils.logger import log


class VisualEmbedder:
    """SOTA visual embeddings using SigLIP for cross-modal search."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initializes the VisualEmbedder with a specific SigLIP model.

        Args:
            model_name: Optional override for the model to use. If None,
                it is read from global settings.
        """
        self.model_name = model_name or settings.siglip_model
        self.device = settings.device
        self._model = None
        self._processor = None
        self._dimension: int | None = None

    def _load_model(self) -> None:
        """Loads the SigLIP model and processor into memory (Lazy Load)."""
        if self._model is not None:
            return

        log(f"[VisualEmbedder] Loading SigLIP: {self.model_name}")

        try:
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
                if self.device == "cuda"
                else torch.float32,
            )
            self._model.to(self.device)
            self._model.eval()

            # Get embedding dimension from a test
            with torch.no_grad():
                dummy = self._processor(
                    text=["test"],
                    return_tensors="pt",
                    padding=True,
                )
                dummy = {k: v.to(self.device) for k, v in dummy.items()}
                out = self._model.get_text_features(**dummy)
                self._dimension = out.shape[-1]

            log(
                f"[VisualEmbedder] Loaded on {self.device}, dim={self._dimension}"
            )

        except Exception as e:
            log(f"[VisualEmbedder] Failed to load: {e}")
            raise

    @property
    def dimension(self) -> int:
        """The output dimension of the visual embeddings (e.g., 1152).

        Returns:
            The embedding dimension as an integer.
        """
        self._load_model()
        return self._dimension or 1152  # SigLIP-SO400M default

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Generates a semantically rich visual embedding for an image.

        Args:
            image_path: Path to the input image file.

        Returns:
            An L2-normalized feature vector (list of floats).
        """
        self._load_model()

        try:
            image = Image.open(image_path).convert("RGB")

            processor = self._processor
            model = self._model
            if processor is None or model is None:
                return [0.0] * self.dimension

            inputs = processor(
                images=image,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = model.get_image_features(**inputs)
                # L2 normalize for cosine similarity
                features = features / features.norm(dim=-1, keepdim=True)

            return features[0].cpu().float().numpy().tolist()

        except Exception as e:
            log(f"[VisualEmbedder] Image embedding failed: {e}")
            # Return zero vector on failure
            return [0.0] * self.dimension

    def embed_text(self, text: str) -> list[float]:
        """Generates a text embedding for cross-modal semantic search.

        Should be used to embed user queries to match against image embeddings.

        Args:
            text: The search query or descriptive text.

        Returns:
            An L2-normalized feature vector (list of floats).
        """
        self._load_model()

        try:
            processor = self._processor
            model = self._model
            if processor is None or model is None:
                return [0.0] * self.dimension

            inputs = processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,  # SigLIP context length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = model.get_text_features(**inputs)
                # L2 normalize for cosine similarity
                features = features / features.norm(dim=-1, keepdim=True)

            return features[0].cpu().float().numpy().tolist()

        except Exception as e:
            log(f"[VisualEmbedder] Text embedding failed: {e}")
            return [0.0] * self.dimension

    def embed_images_batch(
        self, image_paths: list[str | Path]
    ) -> list[list[float]]:
        """Processes multiple images in a single batch to improve throughput.

        Args:
            image_paths: A list of paths to the image files.

        Returns:
            A list of feature vectors, one for each successfully processed image.
        """
        self._load_model()

        results = []
        batch_size = 8  # Process 8 images at a time

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    log(f"[VisualEmbedder] Failed to load {path}: {e}")
                    images.append(Image.new("RGB", (384, 384)))  # Placeholder

            processor = self._processor
            model = self._model
            if processor is None or model is None:
                return []

            inputs = processor(
                images=images,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            for j in range(len(images)):
                results.append(features[j].cpu().float().numpy().tolist())

        return results

    def unload(self) -> None:
        """Releases the SigLIP model and processor from GPU/system memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log("[VisualEmbedder] Unloaded")


# Singleton instance
_visual_embedder: VisualEmbedder | None = None


def get_visual_embedder() -> VisualEmbedder:
    """Retrieves the singleton VisualEmbedder instance.

    Returns:
        The initialized VisualEmbedder.
    """
    global _visual_embedder
    if _visual_embedder is None:
        _visual_embedder = VisualEmbedder()
    return _visual_embedder
