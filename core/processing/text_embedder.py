"""Text embedding service using SentenceTransformers.

Handles loading and inference of text embedding models.
Separated from VectorDB to adhere to SRP.
"""

from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer

from config import settings
from core.utils.hardware import select_embedding_model
from core.utils.logger import log
from core.utils.resource_arbiter import RESOURCE_ARBITER

# Auto-select embedding model based on available VRAM
if settings.embedding_model_override:
    _SELECTED_MODEL = settings.embedding_model_override
else:
    _SELECTED_MODEL, _ = select_embedding_model()


class TextEmbedder:
    """Handles text embedding using SentenceTransformers."""

    VECTOR_SIZE = settings.text_embedding_dim
    # Fix for BGE-M3
    if (
        "bge-m3" in _SELECTED_MODEL.lower()
        or "mxbai" in _SELECTED_MODEL.lower()
    ):
        VECTOR_SIZE = 1024

    def __init__(self, device: str | None = None) -> None:
        """Initialize TextEmbedder.

        Args:
            device: Device to run on. Auto-detected if None.
        """
        self.model_name = _SELECTED_MODEL
        self._device = device
        self._encoder: SentenceTransformer | None = None
        self._init_lock = asyncio.Lock()
        self._last_used = 0.0

    async def encode_texts(
        self, texts: list[str], is_query: bool = False, instruction: str = ""
    ) -> list[list[float]]:
        """Generates embeddings for a list of texts.

        Args:
            texts: A list of strings to embed.
            is_query: Whether these texts are search queries (affects prefixes).
            instruction: Optional instruction specific to the model.

        Returns:
            A list of embeddings (lists of floats).
        """
        if not texts:
            return []

        # Ensure model is loaded
        await self._ensure_model_loaded()

        if not self._encoder:
            raise RuntimeError("Embedding model failed to load")

        # Apply Prefixes based on model type
        processing_texts = texts
        model_lower = self.model_name.lower()

        if "e5" in model_lower:
            prefix = "query: " if is_query else "passage: "
            processing_texts = [prefix + t for t in texts]
        elif "nv-embed-v2" in model_lower:
            if is_query:
                prefix = (
                    instruction
                    or "Instruction: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
                )
                processing_texts = [prefix + t for t in texts]
        elif "mxbai" in model_lower:
            if is_query:
                prefix = (
                    instruction
                    or "Represent this sentence for searching relevant passages: "
                )
                processing_texts = [prefix + t for t in texts]

        # Encode
        try:
            # Use arbiter if on GPU
            if self._encoder.device.type == "cuda":
                async with RESOURCE_ARBITER.acquire("embedding", vram_gb=1.0):
                    loop = asyncio.get_running_loop()
                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self._encoder.encode(
                            processing_texts,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                        ),
                    )
            else:
                # CPU
                loop = asyncio.get_running_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self._encoder.encode(
                        processing_texts,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    ),
                )

            return [e.tolist() for e in embeddings]

        except Exception as e:
            log(f"[TextEmbedder] Encoding failed: {e}")
            raise

    async def encode_text(self, text: str) -> list[float]:
        """Generates embedding for a single text.

        Args:
            text: Input string.

        Returns:
            Embedding vector.
        """
        results = await self.encode_texts([text])
        return results[0] if results else []

    async def _ensure_model_loaded(self) -> None:
        """Lazy loads the model."""
        if self._encoder is not None:
            # Update usage time?
            return

        async with self._init_lock:
            if self._encoder is not None:
                return

            log(f"[TextEmbedder] Loading model: {self.model_name}...")
            try:
                import torch

                device = self._device
                if not device:
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                self._encoder = SentenceTransformer(
                    self.model_name, trust_remote_code=True, device=device
                )
                log(f"[TextEmbedder] Loaded on {device}")

            except Exception as e:
                log(f"[TextEmbedder] Load failed: {e}")
                # Try fallback?
                raise

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._encoder:
            del self._encoder
            self._encoder = None
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            log("[TextEmbedder] Model unloaded")


# Global instance
text_embedder = TextEmbedder()
