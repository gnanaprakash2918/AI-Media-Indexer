"""HITL Feedback Manager for search quality improvement.

This module implements a feedback system that uses past user ratings
to boost or penalize search results. This is a key component of the
Human-in-the-Loop (HITL) search improvement pipeline.

Usage:
    from core.retrieval.hitl_feedback import HITLFeedbackManager

    hitl = HITLFeedbackManager()
    boost = hitl.get_boost_factor(query, video_path, timestamp)
    final_score = original_score * boost
"""

from __future__ import annotations

import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

from core.utils.logger import get_logger

if TYPE_CHECKING:
    from typing import Any

log = get_logger(__name__)


class HITLFeedbackManager:
    """Manages user feedback to improve search ranking.

    This class loads feedback from JSON files stored by the search feedback
    endpoint and provides score boost/penalty factors based on past ratings.

    Feedback files are stored in logs/search_feedback/ with structure:
    {
        "query": "original search query",
        "result_id": "qdrant point ID",
        "video_path": "/path/to/video.mp4",
        "timestamp": 45.2,
        "is_relevant": true/false,
        "feedback_type": "thumbs_up" | "thumbs_down" | "correction",
        "rating": 1-5 (optional),
        "correction": "what should have matched" (optional)
    }
    """

    def __init__(
        self,
        feedback_dir: Path | str = Path("logs/search_feedback"),
        cache_ttl_seconds: int = 300,  # Reload cache every 5 minutes
        positive_boost: float | None = None,  # Use settings if None
        negative_penalty: float | None = None,  # Use settings if None
        max_boost: float | None = None,  # Use settings if None
        min_penalty: float | None = None,  # Use settings if None
    ):
        """Initialize the feedback manager.

        Args:
            feedback_dir: Directory containing feedback JSON files.
            cache_ttl_seconds: How long to cache feedback before reloading.
            positive_boost: Multiplier for positive feedback (>1.0 = boost).
            negative_penalty: Multiplier for negative feedback (<1.0 = penalty).
            max_boost: Maximum cumulative boost allowed.
            min_penalty: Minimum cumulative score multiplier.
        """
        from config import settings

        self.feedback_dir = Path(feedback_dir)
        self.cache_ttl = cache_ttl_seconds
        # Use settings as defaults, allow override
        self.positive_boost = (
            positive_boost
            if positive_boost is not None
            else settings.search_hitl_positive_boost
        )
        self.negative_penalty = (
            negative_penalty
            if negative_penalty is not None
            else settings.search_hitl_negative_penalty
        )
        self.max_boost = (
            max_boost
            if max_boost is not None
            else settings.search_hitl_max_boost
        )
        self.min_penalty = (
            min_penalty
            if min_penalty is not None
            else settings.search_hitl_min_penalty
        )
        self._cache: dict[str, list[dict]] | None = None
        self._cache_time: float = 0
        self._query_embeddings_cache: dict[str, list[float]] = {}

    def _load_feedback(self) -> dict[str, list[dict]]:
        """Load all feedback into memory cache.

        Returns:
            Dict mapping lowercase queries to list of feedback items.
        """
        import time

        # Check if cache is still valid
        if self._cache is not None:
            if time.time() - self._cache_time < self.cache_ttl:
                return self._cache

        feedback: dict[str, list[dict]] = defaultdict(list)

        if not self.feedback_dir.exists():
            log.debug("[HITL] No feedback directory found")
            return feedback

        loaded = 0
        for f in self.feedback_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                query = data.get("query", "").lower().strip()
                if query:
                    feedback[query].append(data)
                    loaded += 1
            except Exception as e:
                log.warning(f"[HITL] Failed to load feedback file {f}: {e}")

        self._cache = dict(feedback)
        self._cache_time = time.time()
        log.info(
            f"[HITL] Loaded {loaded} feedback items for {len(feedback)} queries"
        )

        return self._cache

    def get_boost_factor(
        self,
        query: str,
        video_path: str,
        timestamp: float,
        temporal_tolerance: float = 10.0,
        similarity_threshold: float = 0.7,
    ) -> float:
        """Get score boost/penalty based on past feedback.

        This method finds relevant past feedback for the given query and result,
        then calculates a multiplicative factor to apply to the score.

        Args:
            query: Current search query.
            video_path: Video path of the result.
            timestamp: Timestamp of the result.
            temporal_tolerance: How close timestamps need to be to match (seconds).
            similarity_threshold: Minimum query similarity to consider feedback.

        Returns:
            Boost factor:
            - 1.0 = neutral (no relevant feedback)
            - > 1.0 = boost (past positive feedback)
            - < 1.0 = penalty (past negative feedback)
        """
        feedback = self._load_feedback()
        if not feedback:
            return 1.0

        query_lower = query.lower().strip()

        # Find similar queries in feedback history
        relevant_feedback: list[dict] = []

        for stored_query, items in feedback.items():
            # Calculate query similarity
            similarity = SequenceMatcher(
                None, query_lower, stored_query
            ).ratio()

            if similarity >= similarity_threshold:
                # Add all feedback items for this similar query
                for item in items:
                    item["_query_similarity"] = similarity
                    relevant_feedback.append(item)

        if not relevant_feedback:
            return 1.0

        # Filter to feedback matching this specific result
        boost = 1.0
        matching_count = 0

        for fb in relevant_feedback:
            fb_video = fb.get("video_path", "")
            fb_ts = fb.get("timestamp", 0)

            # Normalize paths for comparison
            fb_video_norm = Path(fb_video).name if fb_video else ""
            video_path_norm = Path(video_path).name if video_path else ""

            # Check for match: same video and within temporal tolerance
            video_match = (
                fb_video_norm == video_path_norm or fb_video == video_path
            )
            temporal_match = abs(fb_ts - timestamp) <= temporal_tolerance

            if video_match and temporal_match:
                matching_count += 1
                query_sim = fb.get("_query_similarity", 1.0)

                if fb.get("is_relevant"):
                    # Positive feedback: apply boost scaled by query similarity
                    # boost = 1.5 + (1.5-1.0)*sim = up to 1.5x for perfect similarity
                    boost_factor = 1.0 + (self.positive_boost - 1.0) * query_sim
                    boost *= boost_factor
                    log.debug(
                        f"[HITL] Positive feedback for {video_path_norm}@{timestamp:.1f}s "
                        f"(query_sim={query_sim:.2f}, boost_factor={boost_factor:.2f}), cumulative={boost:.2f}"
                    )
                else:
                    # Negative feedback: apply penalty scaled by query similarity
                    # penalty = 0.5 + (1.0-0.5)*(1-sim) = 0.5 for perfect similarity
                    penalty_factor = self.negative_penalty + (
                        1.0 - self.negative_penalty
                    ) * (1.0 - query_sim)
                    boost *= penalty_factor
                    log.debug(
                        f"[HITL] Negative feedback for {video_path_norm}@{timestamp:.1f}s "
                        f"(query_sim={query_sim:.2f}, penalty={penalty_factor:.2f}), cumulative={boost:.2f}"
                    )

        if matching_count > 0:
            log.info(
                f"[HITL] Applied {matching_count} feedback items, final boost={boost:.2f}"
            )

        # Clamp to configured range
        return max(self.min_penalty, min(self.max_boost, boost))

    def get_all_feedback_for_query(
        self,
        query: str,
        similarity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Get all feedback items relevant to a query.

        Useful for debugging and analysis.

        Args:
            query: Search query to find feedback for.
            similarity_threshold: Minimum query similarity.

        Returns:
            List of feedback items with added _query_similarity field.
        """
        feedback = self._load_feedback()
        query_lower = query.lower().strip()

        relevant: list[dict] = []
        for stored_query, items in feedback.items():
            similarity = SequenceMatcher(
                None, query_lower, stored_query
            ).ratio()
            if similarity >= similarity_threshold:
                for item in items:
                    item_copy = item.copy()
                    item_copy["_query_similarity"] = similarity
                    item_copy["_stored_query"] = stored_query
                    relevant.append(item_copy)

        # Sort by similarity descending
        relevant.sort(key=lambda x: x.get("_query_similarity", 0), reverse=True)
        return relevant

    def get_feedback_stats(self) -> dict[str, Any]:
        """Get overall statistics on collected feedback.

        Returns:
            Dictionary with stats like total count, positive ratio, etc.
        """
        feedback = self._load_feedback()

        total = sum(len(items) for items in feedback.values())
        positive = sum(
            1
            for items in feedback.values()
            for item in items
            if item.get("is_relevant")
        )
        negative = total - positive

        return {
            "total_feedback": total,
            "unique_queries": len(feedback),
            "positive_count": positive,
            "negative_count": negative,
            "positive_ratio": positive / total if total > 0 else None,
            "feedback_dir": str(self.feedback_dir),
        }

    def clear_cache(self) -> None:
        """Force reload of feedback on next access."""
        self._cache = None
        self._cache_time = 0


# Module-level singleton for convenience
_default_manager: HITLFeedbackManager | None = None


def get_hitl_manager() -> HITLFeedbackManager:
    """Get the default HITL feedback manager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = HITLFeedbackManager()
    return _default_manager
