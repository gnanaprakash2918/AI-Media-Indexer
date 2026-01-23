"""Late Interaction (ColBERT-style) Retrieval.

Implements multi-vector retrieval using MaxSim (Late Interaction) scoring.
This allows for fine-grained matching between query terms and document tokens,
providing superior accuracy over single-vector dense retrieval.

Uses BGE-M3 (BAAI/bge-m3) which supports:
1. Dense Retrieval (CLS embedding)
2. Sparse Retrieval (Splade-like)
3. Multi-Vector / ColBERT (Token embeddings)

User Priority: ACCURACY.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core.utils.logger import get_logger

log = get_logger(__name__)


class ColBERTRetriever:
    """ColBERT-style Late Interaction Retriever using BGE-M3.

    Performs multi-vector retrieval where every query token interacts with
    every document token (MaxSim), providing deep semantic matching.
    """

    def __init__(self, device: str | None = None):
        self._device = device
        self._model = None
        self._load_lock = asyncio.Lock()
        self._load_failed = False

    def _get_device(self) -> str:
        if self._device:
            return self._device
        return "cuda" if torch.cuda.is_available() else "cpu"

    async def _lazy_load(self) -> bool:
        """Load BGE-M3 model lazily."""
        if self._model is not None:
            return True

        if self._load_failed:
            return False

        async with self._load_lock:
            if self._model is not None:
                return True

            try:
                log.info("[ColBERT] Loading BGE-M3 for late interaction...")
                # We use FlagEmbedding for BGE-M3 specifically as it handles the multi-vector logic well
                from FlagEmbedding import BGEM3FlagModel

                device = self._get_device()

                # Load with FP16 for efficiency since this model is large
                self._model = BGEM3FlagModel(
                    "BAAI/bge-m3", use_fp16=True, device=device
                )

                self._device = device
                log.info(f"[ColBERT] Loaded BGE-M3 on {device}")
                return True

            except ImportError:
                log.warning(
                    "[ColBERT] FlagEmbedding not installed. Install with `pip install FlagEmbedding`"
                )
                self._load_failed = True
                return False
            except Exception as e:
                log.error(f"[ColBERT] Failed to load BGE-M3: {e}")
                self._load_failed = True
                return False

    async def encode_query(self, query: str) -> dict[str, Any] | None:
        """Encode query into dense, sparse, and multi-vector embeddings.

        Args:
            query: Search query

        Returns:
            Dict containing:
                - dense_vecs: Global embedding
                - lexial_weights: Sparse weights
                - colbert_vecs: Multi-vector token embeddings
        """
        if not await self._lazy_load():
            return None

        try:
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    query,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True,
                ),
            )
            return output
        except Exception as e:
            log.error(f"[ColBERT] Query encoding failed: {e}")
            return None

    async def encode_documents(
        self, documents: list[str]
    ) -> list[dict[str, Any]]:
        """Encode documents for multi-vector retrieval.

        Returns list of embeddings dicts.
        """
        if not await self._lazy_load():
            return []

        try:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    documents,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True,
                ),
            )

            # encode returns a dict of arrays if input is list
            # We need to restructure into list of dicts for easier handling
            results = []

            # The output format depends on the library version, but usually:
            # output['dense_vecs'] -> valid array
            # output['colbert_vecs'] -> list of arrays (variable length)

            dense = output["dense_vecs"]
            sparse = output["lexical_weights"]
            colbert = output["colbert_vecs"]

            for i in range(len(documents)):
                results.append(
                    {
                        "dense_vecs": dense[i],
                        "lexical_weights": sparse[i],
                        "colbert_vecs": colbert[i],
                    }
                )

            return results

        except Exception as e:
            log.error(f"[ColBERT] Document encoding failed: {e}")
            return []

    def compute_score(
        self, query_embeddings: np.ndarray, doc_embeddings: np.ndarray
    ) -> float:
        """Compute MaxSim (Late Interaction) score.

        Score = Sum_over_query_tokens( Max_over_doc_tokens( dot(q_i, d_j) ) )

        Args:
            query_embeddings: Shape (num_query_tokens, dim) or (dim,)
            doc_embeddings: Shape (num_doc_tokens, dim) or (dim,)

        Returns:
            Scalar score
        """
        if query_embeddings is None or doc_embeddings is None:
            return 0.0

        try:
            # Ensure 2D shape: (num_tokens, dim)
            # Handle case where embeddings are 1D (single token)
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            if doc_embeddings.ndim == 1:
                doc_embeddings = doc_embeddings.reshape(1, -1)

            # Validate shapes after reshaping
            if query_embeddings.ndim != 2 or doc_embeddings.ndim != 2:
                log.error(
                    f"[ColBERT] Invalid embedding dimensions: "
                    f"query={query_embeddings.shape}, doc={doc_embeddings.shape}"
                )
                return 0.0

            # Validate non-empty
            if query_embeddings.shape[0] == 0 or doc_embeddings.shape[0] == 0:
                log.warning("[ColBERT] Empty embeddings provided")
                return 0.0

            # Convert to torch for efficient matrix ops
            q = torch.from_numpy(query_embeddings).to(self._device)
            d = torch.from_numpy(doc_embeddings).to(self._device)

            # Normalize vectors if not already normalized (BGE-M3 colbert vectors are usually normalized)
            # But verifying doesn't hurt
            # q = F.normalize(q, p=2, dim=1)
            # d = F.normalize(d, p=2, dim=1)

            # Sim matrix: (num_q, num_d)
            sim_matrix = torch.matmul(q, d.T)

            # Max over document tokens (dim=1 means max over columns)
            max_sim_values, _ = torch.max(sim_matrix, dim=1)

            # Sum over query tokens
            score = torch.sum(max_sim_values)

            return float(score.item())

        except Exception as e:
            log.error(
                f"[ColBERT] Scoring failed: {e}, "
                f"query_shape={query_embeddings.shape if hasattr(query_embeddings, 'shape') else 'unknown'}, "
                f"doc_shape={doc_embeddings.shape if hasattr(doc_embeddings, 'shape') else 'unknown'}"
            )
            return 0.0
