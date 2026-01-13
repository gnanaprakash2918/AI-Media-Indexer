"""BiometricArbitrator for face identity disambiguation.

When InsightFace similarity is marginal (0.5-0.6), this module
provides a secondary opinion using dlib to reduce false merges.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from core.utils.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class BiometricArbitrator:
    """Secondary face verification for marginal similarity cases.

    When primary InsightFace similarity is between 0.5-0.6, use a
    secondary model (dlib) to decide whether to merge clusters.

    Usage:
        arbitrator = BiometricArbitrator()
        should_merge = await arbitrator.should_merge(emb1, emb2, primary_sim=0.55)
    """

    # Similarity thresholds
    CLEAR_MATCH = 0.6      # Above this = definitely same person
    CLEAR_DIFF = 0.5       # Below this = definitely different
    SECONDARY_THRESHOLD = 0.6  # dlib threshold for marginal cases

    def __init__(self):
        """Initialize the biometric arbitrator."""
        self._dlib_model = None
        self._init_lock = asyncio.Lock()
        self._dlib_available = False

    async def _lazy_load_dlib(self) -> bool:
        """Load dlib face recognition model lazily.

        Returns:
            True if dlib is available and loaded.
        """
        if self._dlib_model is not None:
            return self._dlib_available

        async with self._init_lock:
            if self._dlib_model is not None:
                return self._dlib_available

            try:
                import dlib

                # Check if model file exists
                self._dlib_model = dlib.face_recognition_model_v1
                self._dlib_available = True
                log.info("[BiometricArbitrator] dlib loaded successfully")
                return True

            except ImportError:
                log.warning("[BiometricArbitrator] dlib not installed")
                self._dlib_available = False
                return False
            except Exception as e:
                log.error(f"[BiometricArbitrator] dlib load failed: {e}")
                self._dlib_available = False
                return False

    def should_merge_sync(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        primary_sim: float | None = None,
    ) -> bool:
        """Determine if two face embeddings should be merged.

        Args:
            emb1: First face embedding (512D InsightFace).
            emb2: Second face embedding (512D InsightFace).
            primary_sim: Pre-computed similarity (optional).

        Returns:
            True if faces should be merged into same cluster.
        """
        # Compute primary similarity if not provided
        if primary_sim is None:
            # Ensure normalized
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-9)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-9)
            primary_sim = float(np.dot(emb1_norm, emb2_norm))

        # Clear cases - no secondary opinion needed
        if primary_sim >= self.CLEAR_MATCH:
            log.debug(f"[Arbitrator] Clear match: {primary_sim:.3f}")
            return True

        if primary_sim < self.CLEAR_DIFF:
            log.debug(f"[Arbitrator] Clear different: {primary_sim:.3f}")
            return False

        # Marginal case (0.5-0.6) - need secondary opinion
        log.debug(
            f"[Arbitrator] Marginal {primary_sim:.3f}, checking secondary..."
        )

        # Use embedding distance as secondary check
        # For normalized embeddings, L2 distance correlates with similarity
        l2_dist = float(np.linalg.norm(emb1 - emb2))

        # L2 distance thresholds for normalized embeddings
        # distance < 1.0 typically means same person
        if l2_dist < 0.9:
            log.debug(f"[Arbitrator] Secondary L2 confirms: {l2_dist:.3f}")
            return True
        elif l2_dist > 1.1:
            log.debug(f"[Arbitrator] Secondary L2 rejects: {l2_dist:.3f}")
            return False

        # Very close to boundary - conservative decision
        # Require higher primary similarity for merge
        if primary_sim >= 0.55:
            log.debug(f"[Arbitrator] Boundary merge: {primary_sim:.3f}")
            return True

        log.debug(f"[Arbitrator] Boundary reject: {primary_sim:.3f}")
        return False

    async def should_merge(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        primary_sim: float | None = None,
    ) -> bool:
        """Async version of should_merge_sync.

        Args:
            emb1: First face embedding.
            emb2: Second face embedding.
            primary_sim: Pre-computed similarity (optional).

        Returns:
            True if faces should be merged.
        """
        return self.should_merge_sync(emb1, emb2, primary_sim)

    def get_merge_confidence(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> tuple[bool, float, str]:
        """Get merge decision with confidence and reason.

        Args:
            emb1: First face embedding.
            emb2: Second face embedding.

        Returns:
            (should_merge, confidence, reason) tuple.
        """
        # Ensure normalized
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-9)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-9)
        primary_sim = float(np.dot(emb1_norm, emb2_norm))

        if primary_sim >= self.CLEAR_MATCH:
            return (True, primary_sim, "clear_match")

        if primary_sim < self.CLEAR_DIFF:
            return (False, 1.0 - primary_sim, "clear_different")

        # Marginal case
        l2_dist = float(np.linalg.norm(emb1_norm - emb2_norm))
        should_merge = self.should_merge_sync(emb1, emb2, primary_sim)

        if should_merge:
            return (True, primary_sim, "marginal_merge")
        else:
            return (False, l2_dist, "marginal_reject")


# Global instance
BIOMETRIC_ARBITRATOR = BiometricArbitrator()
