"""Text embedding encoder lifecycle management.

Handles lazy loading, caching, VRAM management, and model prefix logic
for SentenceTransformer models.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from config import settings
from core.storage.constants import SELECTED_MODEL
from core.utils.logger import log
from core.utils.observe import observe


class TextEncoder:
    """Manages the SentenceTransformer embedding model lifecycle."""

    def __init__(self) -> None:
        self.model_name = SELECTED_MODEL
        self.encoder: SentenceTransformer | None = None
        self._encoder_last_used: float = 0.0
        self._idle_unload_seconds = getattr(settings, "encoder_idle_timeout", 300)

        self._embedding_cache: OrderedDict = OrderedDict()
        self._embedding_cache_max_size = getattr(
            settings, "embedding_cache_size", 1000
        )

    def _load_model(self) -> SentenceTransformer:
        """Load SentenceTransformer from local cache or HuggingFace Hub."""
        models_dir = settings.model_cache_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        local_model_dir = models_dir / self.model_name
        target_device = settings.device or "cpu"

        def _create(path_or_name: str, device: str) -> SentenceTransformer:
            log(
                "Creating SentenceTransformer",
                path_or_name=path_or_name,
                device=device,
            )
            return SentenceTransformer(
                path_or_name,
                device=device,
                trust_remote_code=True,
            )

        if local_model_dir.exists():
            log(
                "Loading cached model",
                path=str(local_model_dir),
                device=target_device,
            )
            try:
                return _create(str(local_model_dir), device=target_device)
            except Exception as exc:
                log(f"GPU Load Failed: {exc}. Retrying on CPU...", level="warning")
                try:
                    return _create(str(local_model_dir), device="cpu")
                except Exception:
                    pass  # Fall through to re-download

        log("Local model missing/corrupt, downloading from Hub", model=self.model_name)

        try:
            snapshot_download(
                repo_id=self.model_name,
                local_dir=str(local_model_dir),
                token=settings.hf_token,
            )
        except Exception as dl_exc:
            log(f"Snapshot Download Failed: {dl_exc}", level="error")
            return _create(self.model_name, device=target_device)

        try:
            return _create(str(local_model_dir), device=target_device)
        except Exception:
            return _create(self.model_name, device=target_device)

    def to_cpu(self) -> None:
        """Move encoder to CPU to free GPU VRAM."""
        if self.encoder is not None:
            try:
                self.encoder = self.encoder.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                log("Encoder moved to CPU, VRAM freed")
            except Exception as e:
                log(f"Failed to move encoder to CPU: {e}", level="WARNING")

    def to_gpu(self) -> None:
        """Move encoder back to configured GPU device."""
        device = settings.device or "cuda"
        if device != "cpu" and self.encoder is not None:
            try:
                self.encoder = self.encoder.to(device)
                log(f"Encoder moved back to {device}")
            except Exception as e:
                log(f"Failed to move encoder to GPU: {e}", level="WARNING")

    async def ensure_loaded(self, job_id: str | None = None) -> None:
        """Load encoder if not already loaded. Manages VRAM via ResourceArbiter."""
        if self.encoder is None:
            vram_gb = 1.0
            model_lower = self.model_name.lower()
            if "nv-embed-v2" in model_lower:
                vram_gb = 16.0
            elif "sfr-embedding-2" in model_lower:
                vram_gb = 4.0
            elif "bge-m3" in model_lower:
                vram_gb = 2.0

            from core.utils.resource_arbiter import RESOURCE_ARBITER

            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    allocated = await RESOURCE_ARBITER.ensure_loaded(
                        "embedding_encoder",
                        vram_gb=vram_gb,
                        cleanup_fn=self.unload,
                    )
                    if not allocated:
                        log(
                            "RESOURCE_ARBITER could not allocate VRAM for encoder, loading anyway",
                            level="WARNING",
                        )
                else:
                    RESOURCE_ARBITER.register_model(
                        "embedding_encoder", self.unload
                    )
            except Exception as e:
                log(
                    f"RESOURCE_ARBITER integration failed, loading anyway: {e}",
                    level="WARNING",
                )
                RESOURCE_ARBITER.register_model("embedding_encoder", self.unload)

            self.encoder = self._load_model()

        self._encoder_last_used = time.time()

    def unload(self) -> None:
        """Unload encoder from memory to free VRAM."""
        if self.encoder is None:
            return
        log("Unloading text encoder to free VRAM")
        del self.encoder
        self.encoder = None
        from core.utils.hardware import cleanup_vram

        cleanup_vram()

    def unload_if_idle(self) -> bool:
        """Unload encoder if idle time exceeded. Returns True if unloaded."""
        if self.encoder is None:
            return False
        idle_time = time.time() - self._encoder_last_used
        if idle_time > self._idle_unload_seconds:
            log(f"Unloading encoder after {idle_time:.0f}s idle")
            self.unload()
            return True
        return False

    @observe("db_encode_texts")
    async def encode_texts(
        self,
        texts: str | list[str],
        batch_size: int = 1,
        show_progress_bar: bool = False,
        is_query: bool = False,
        job_id: str | None = None,
    ) -> list[list[float]]:
        """Transform text(s) into vector embeddings with caching."""
        await self.ensure_loaded(job_id=job_id)

        if isinstance(texts, str):
            input_texts = [texts]
        else:
            input_texts = list(texts)

        # Check cache
        indices_to_compute = []
        cache_keys = [(t, is_query, self.model_name) for t in input_texts]
        results: list[Any] = [None] * len(input_texts)

        for i, key in enumerate(cache_keys):
            if key in self._embedding_cache:
                self._embedding_cache.move_to_end(key)
                results[i] = self._embedding_cache[key]
            else:
                indices_to_compute.append(i)

        if not indices_to_compute:
            return [list(r) for r in results]

        texts_to_compute = [input_texts[i] for i in indices_to_compute]

        # Apply model-specific prefixes
        model_lower = self.model_name.lower()
        processed_texts = texts_to_compute

        if "e5" in model_lower:
            prefix = "query: " if is_query else "passage: "
            processed_texts = [prefix + t for t in texts_to_compute]
        elif "nv-embed-v2" in model_lower:
            if is_query:
                prefix = "Instruction: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
                processed_texts = [prefix + t for t in texts_to_compute]
        elif "mxbai" in model_lower:
            if is_query:
                prefix = "Represent this sentence for searching relevant passages: "
                processed_texts = [prefix + t for t in texts_to_compute]

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if self.encoder is None:
            await self.ensure_loaded(job_id=job_id)
            if self.encoder is None:
                log("Encoder not available", level="ERROR")
                return []

        embeddings = self.encoder.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        computed_list = [list(e) for e in embeddings]
        for idx_in_batch, original_idx in enumerate(indices_to_compute):
            emb = computed_list[idx_in_batch]
            results[original_idx] = emb

            key = cache_keys[original_idx]
            self._embedding_cache[key] = emb

            if len(self._embedding_cache) > self._embedding_cache_max_size:
                self._embedding_cache.popitem(last=False)

        return results

    async def get_embedding(self, text: str) -> list[float] | None:
        """Get a single query embedding."""
        try:
            return (await self.encode_texts(text, is_query=True))[0]
        except Exception as e:
            log(
                f"Error generating embedding for '{text[:20]}...': {e}",
                level="ERROR",
            )
            return None
