"""Advanced Embedding Models: NV-Embed-v2, Nomic, and more.

High-accuracy embedding models from the Embedding Council.
User priority: ACCURACY over storage/speed.

Based on Research (Part 7 - Embedding Council):
- NV-Embed-v2: 7B params, SOTA on 72 benchmarks - INCLUDED for max accuracy
- Nomic-v1.5: 8192 context, Matryoshka embeddings
- BGE-M3: Hybrid dense/sparse
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class NVEmbedEncoder:
    """NV-Embed-v2: State-of-the-art embedding model.

    7B parameters, leads 72 benchmarks on MTEB.
    User prioritized ACCURACY over compute cost.

    Usage:
        encoder = NVEmbedEncoder()
        embedding = await encoder.encode_text("search query")
        embedding = await encoder.encode_query("What is AI?")
    """

    def __init__(self, device: str | None = None):
        """Initialize NV-Embed-v2 encoder.

        Args:
            device: Device to run on. Auto-detected if None.
        """
        self._device = device
        self._model = None
        self._tokenizer = None
        self._init_lock = asyncio.Lock()
        self._load_failed = False

    def _get_device(self) -> str:
        """Get device to use."""
        if self._device:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    async def _lazy_load(self) -> bool:
        """Load NV-Embed-v2 model lazily."""
        if self._model is not None:
            return True

        if self._load_failed:
            return False

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info(
                    "[NV-Embed-v2] Loading SOTA embedding model (7B params)..."
                )

                import torch
                from transformers import AutoModel, AutoTokenizer

                model_id = "nvidia/NV-Embed-v2"

                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,  # Use FP16 for memory efficiency
                    trust_remote_code=True,
                )

                device = self._get_device()
                self._model.to(device)
                self._model.eval()
                self._device = device

                log.info(f"[NV-Embed-v2] Loaded on {device}")
                return True

            except ImportError as e:
                log.warning(f"[NV-Embed-v2] Dependencies not available: {e}")
                self._load_failed = True
                return False
            except Exception as e:
                log.warning(f"[NV-Embed-v2] Load failed: {e}")
                self._load_failed = True
                return False

    async def encode_text(
        self,
        text: str,
        instruction: str = "",
    ) -> np.ndarray | None:
        """Encode text to embedding.

        Args:
            text: Text to encode.
            instruction: Optional instruction prefix for task-specific encoding.

        Returns:
            4096-dimensional embedding.
        """
        if not await self._lazy_load():
            return None

        try:
            import torch

            # NV-Embed uses instruction-based encoding
            if instruction:
                full_text = f"{instruction}\n{text}"
            else:
                full_text = text

            inputs = self._tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use pooled output or mean pooling
                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    embedding = outputs.pooler_output
                else:
                    embedding = outputs.last_hidden_state.mean(dim=1)

            emb = embedding.cpu().numpy().flatten()

            # Normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            return emb

        except Exception as e:
            log.error(f"[NV-Embed-v2] Encoding failed: {e}")
            return None

    async def encode_query(self, query: str) -> np.ndarray | None:
        """Encode a search query.

        Uses query-specific instruction for better retrieval.

        Args:
            query: Search query.

        Returns:
            Query embedding.
        """
        return await self.encode_text(
            query,
            instruction="Instruct: Given a query, retrieve relevant passages\nQuery: ",
        )

    async def encode_document(self, document: str) -> np.ndarray | None:
        """Encode a document/passage.

        Args:
            document: Document text.

        Returns:
            Document embedding.
        """
        return await self.encode_text(document)

    async def encode_batch(
        self,
        texts: list[str],
        instruction: str = "",
    ) -> list[np.ndarray]:
        """Encode multiple texts efficiently.

        Args:
            texts: List of texts.
            instruction: Optional instruction prefix.

        Returns:
            List of embeddings.
        """
        embeddings = []
        for text in texts:
            emb = await self.encode_text(text, instruction)
            if emb is not None:
                embeddings.append(emb)
        return embeddings

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        log.info("[NV-Embed-v2] Resources released")


class NomicEmbedEncoder:
    """Nomic-v1.5: Long context embeddings with Matryoshka support.

    8192 token context - perfect for long transcripts.
    Matryoshka: Can use 64-768 dims adaptively.

    Usage:
        encoder = NomicEmbedEncoder()
        embedding = await encoder.encode("long transcript text...")
    """

    def __init__(self, device: str | None = None):
        """Initialize Nomic encoder.

        Args:
            device: Device to run on.
        """
        self._device = device
        self._model = None
        self._tokenizer = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load Nomic model lazily."""
        if self._model is not None:
            return True

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info("[Nomic] Loading nomic-embed-text-v1.5...")

                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    "nomic-ai/nomic-embed-text-v1.5",
                    trust_remote_code=True,
                )

                import torch

                device = self._device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._model.to(device)
                self._device = device

                log.info(f"[Nomic] Loaded on {device}")
                return True

            except Exception as e:
                log.error(f"[Nomic] Load failed: {e}")
                return False

    async def encode(
        self,
        text: str,
        task: str = "search_document",
        matryoshka_dim: int | None = None,
    ) -> np.ndarray | None:
        """Encode text with Nomic.

        Args:
            text: Text to encode.
            task: Task prefix ('search_document', 'search_query', 'classification').
            matryoshka_dim: Optional dimension to truncate to (64, 128, 256, 512, 768).

        Returns:
            Text embedding.
        """
        if not await self._lazy_load():
            return None

        try:
            # Add task prefix
            prefixed = f"{task}: {text}"

            embedding = self._model.encode(prefixed, convert_to_numpy=True)

            # Matryoshka: truncate if requested
            if matryoshka_dim and matryoshka_dim < len(embedding):
                embedding = embedding[:matryoshka_dim]

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            log.error(f"[Nomic] Encoding failed: {e}")
            return None

    async def encode_query(self, query: str) -> np.ndarray | None:
        """Encode a search query."""
        return await self.encode(query, task="search_query")

    async def encode_document(self, document: str) -> np.ndarray | None:
        """Encode a document/transcript."""
        return await self.encode(document, task="search_document")

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        log.info("[Nomic] Resources released")


class EmbeddingEnsemble:
    """Ensemble of embeddings for maximum accuracy.

    Combines multiple models for best-in-class retrieval:
    - NV-Embed-v2: SOTA accuracy
    - Nomic: Long context
    - BGE-M3: Hybrid search
    """

    def __init__(
        self,
        use_nv_embed: bool = True,
        use_nomic: bool = True,
        use_bge: bool = True,
        device: str | None = None,
    ):
        """Initialize embedding ensemble.

        Args:
            use_nv_embed: Include NV-Embed-v2 (heavy but accurate).
            use_nomic: Include Nomic (long context).
            use_bge: Include BGE-M3 (hybrid).
            device: Device to run on.
        """
        self._device = device
        self._nv_embed = NVEmbedEncoder(device) if use_nv_embed else None
        self._nomic = NomicEmbedEncoder(device) if use_nomic else None
        self._bge = None  # BGE is typically used via existing infrastructure
        self._use_bge = use_bge

    async def encode_query(self, query: str) -> dict[str, np.ndarray]:
        """Encode query with all available models.

        Args:
            query: Search query.

        Returns:
            Dict of {model_name: embedding}.
        """
        embeddings = {}

        if self._nv_embed:
            emb = await self._nv_embed.encode_query(query)
            if emb is not None:
                embeddings["nv_embed_v2"] = emb

        if self._nomic:
            emb = await self._nomic.encode_query(query)
            if emb is not None:
                embeddings["nomic"] = emb

        return embeddings

    async def encode_document(self, document: str) -> dict[str, np.ndarray]:
        """Encode document with all available models.

        Args:
            document: Document text.

        Returns:
            Dict of {model_name: embedding}.
        """
        embeddings = {}

        if self._nv_embed:
            emb = await self._nv_embed.encode_document(document)
            if emb is not None:
                embeddings["nv_embed_v2"] = emb

        if self._nomic:
            emb = await self._nomic.encode_document(document)
            if emb is not None:
                embeddings["nomic"] = emb

        return embeddings

    def cleanup(self) -> None:
        """Release all resources."""
        if self._nv_embed:
            self._nv_embed.cleanup()
        if self._nomic:
            self._nomic.cleanup()
        log.info("[EmbeddingEnsemble] Resources released")
