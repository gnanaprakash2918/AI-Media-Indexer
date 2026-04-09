"""Prompt Injection Defense Layer using dynamic semantic pattern matching."""

import math
from typing import TYPE_CHECKING

from core.utils.logger import log

if TYPE_CHECKING:
    from core.storage.db import VectorDB


class QuerySanitizer:
    """Detects adversarial logic via dynamic heuristics and semantic embedding similarity.

    Replaces brittle hardcoded regexes with dynamic vector space comparisons
    and topological character analysis.
    """

    def __init__(self):
        from config import settings

        # Evolving list of adversarial intents inside the latent space
        self.adversarial_intents = settings.security_adversarial_intents
        # Benign structural baselines for relative semantic calibration
        self.benign_baselines = settings.security_benign_baselines

        self._adversarial_embeddings: list[list[float]] | None = None
        self._benign_embeddings: list[list[float]] | None = None

        # Dynamic semantic margin instead of a hardcoded 0.82
        self.dynamic_margin_threshold = (
            settings.security_dynamic_margin_threshold
        )

    async def _init_embeddings(self, db: "VectorDB") -> None:
        """Dynamically load and cache embeddings for relative calculation."""
        if self._adversarial_embeddings is None:
            log("[Security] Provisioning relative intent cluster embeddings...")
            self._adversarial_embeddings = await db.encode_texts(
                self.adversarial_intents, is_query=False
            )
            self._benign_embeddings = await db.encode_texts(
                self.benign_baselines, is_query=False
            )

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate fast cosine similarity."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    async def sanitize(self, query: str, db: "VectorDB") -> tuple[bool, str]:
        """Validates query against dynamic heuristics and relative semantic clusters."""
        if len(query) > 1500:
            return (
                False,
                "Security constraint violated: Query exceeds physical context boundaries.",
            )

        special_chars = sum(
            1 for c in query if not c.isalnum() and not c.isspace()
        )
        if special_chars / max(len(query), 1) > 0.4:
            return (
                False,
                "Security constraint violated: Query features anomalous topological density.",
            )

        try:
            await self._init_embeddings(db)
            query_emb = await db.get_embedding(query)

            if (
                query_emb
                and self._adversarial_embeddings
                and self._benign_embeddings
            ):
                # Calculate max similarity to adversarial space
                max_adv_sim = max(
                    self._cosine_similarity(query_emb, adv_emb)
                    for adv_emb in self._adversarial_embeddings
                )

                # Calculate max similarity to benign space for relative baseline
                max_benign_sim = max(
                    self._cosine_similarity(query_emb, ben_emb)
                    for ben_emb in self._benign_embeddings
                )

                # DYNAMIC PER-QUERY BASELINE:
                # If adversarial distance outweighs benign baseline by margin X, trigger intercept
                semantic_delta = max_adv_sim - max_benign_sim

                if semantic_delta > self.dynamic_margin_threshold:
                    log(
                        f"[Security] Intercepted semantic attack vector! (Adv: {max_adv_sim:.2f}, Benign: {max_benign_sim:.2f}, Delta: {semantic_delta:.2f})"
                    )
                    return (
                        False,
                        f"Security constraint violated: Query semantics logically align with restricted adversarial patterns (Delta {semantic_delta:.2f}).",
                    )
        except Exception as e:
            log(f"[Security] Dynamic semantic check degraded gracefully: {e}")

        return True, "Passed"


# Singleton instance
query_sanitizer = QuerySanitizer()
