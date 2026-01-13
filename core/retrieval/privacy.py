"""PrivacyFilter for personal/movie media mode filtering.

Ensures personal faces/voices are excluded from agent search
and movie content requires confirmation before external lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from core.utils.logger import get_logger

if TYPE_CHECKING:
    from core.storage.db import VectorDB

log = get_logger(__name__)


MediaMode = Literal["personal", "movie", "all"]


@dataclass
class SearchResult:
    """A search result item."""

    id: str
    score: float
    timestamp: float | None = None
    cluster_id: str | None = None
    media_path: str | None = None
    description: str | None = None
    needs_confirmation: bool = False
    payload: dict = field(default_factory=dict)


class PrivacyFilter:
    """Filter search results based on privacy mode.

    Privacy Modes:
    - personal: Only return HITL-approved faces/voices, no external search
    - movie: Allow all but mark for confirmation before agent/external use
    - all: No filtering (for internal/admin use only)

    Usage:
        filter = PrivacyFilter(db)
        filtered = await filter.filter_results(results, mode="personal", approved={"cluster_1"})
    """

    def __init__(self, db: "VectorDB | None" = None):
        """Initialize the privacy filter.

        Args:
            db: Optional VectorDB instance for cluster lookups.
        """
        self.db = db
        self._approved_clusters: set[str] = set()

    def set_approved_clusters(self, cluster_ids: set[str]) -> None:
        """Set the HITL-approved cluster IDs.

        Args:
            cluster_ids: Set of approved cluster IDs.
        """
        self._approved_clusters = cluster_ids
        log.debug(f"[PrivacyFilter] Set {len(cluster_ids)} approved clusters")

    async def filter_results(
        self,
        results: list[SearchResult],
        mode: MediaMode = "personal",
        hitl_approved: set[str] | None = None,
    ) -> list[SearchResult]:
        """Filter results based on privacy mode.

        Args:
            results: List of search results to filter.
            mode: Privacy mode - 'personal', 'movie', or 'all'.
            hitl_approved: Optional set of HITL-approved cluster IDs.

        Returns:
            Filtered list of search results.
        """
        if hitl_approved is None:
            hitl_approved = self._approved_clusters

        if mode == "personal":
            # Only return HITL-approved faces/voices
            filtered = []
            for r in results:
                if r.cluster_id is None or r.cluster_id in hitl_approved:
                    filtered.append(r)
                else:
                    log.debug(
                        f"[PrivacyFilter] Excluding non-approved: {r.cluster_id}"
                    )
            log.info(
                f"[PrivacyFilter] Personal mode: {len(filtered)}/{len(results)}"
            )
            return filtered

        elif mode == "movie":
            # Return all, but mark as needs_confirmation for agent use
            for r in results:
                r.needs_confirmation = True
            log.info(
                f"[PrivacyFilter] Movie mode: {len(results)} marked for confirmation"
            )
            return results

        else:  # mode == "all"
            log.debug("[PrivacyFilter] All mode: no filtering")
            return results

    def is_personal_content(self, media_path: str) -> bool:
        """Check if media is personal content (not movie/public).

        Uses path heuristics and metadata to determine content type.

        Args:
            media_path: Path to the media file.

        Returns:
            True if content appears to be personal.
        """
        # Simple heuristics - can be expanded
        personal_indicators = [
            "personal",
            "family",
            "home",
            "private",
            "dcim",
            "camera",
            "photos",
            "videos",
        ]
        path_lower = media_path.lower()

        for indicator in personal_indicators:
            if indicator in path_lower:
                log.debug(f"[PrivacyFilter] Personal content: {indicator}")
                return True

        movie_indicators = [
            "movie",
            "film",
            "series",
            "episode",
            "season",
            "1080p",
            "720p",
            "bluray",
            "dvdrip",
        ]

        for indicator in movie_indicators:
            if indicator in path_lower:
                return False

        # Default: treat as personal (safer)
        return True

    async def should_allow_external_search(
        self,
        cluster_id: str,
        mode: MediaMode,
        user_confirmed: bool = False,
    ) -> bool:
        """Check if external search (TMDB, Brave) is allowed.

        Args:
            cluster_id: The cluster ID to search.
            mode: Current privacy mode.
            user_confirmed: Whether user has confirmed this search.

        Returns:
            True if external search is allowed.
        """
        if mode == "personal":
            # Never allow for personal content
            log.warning(
                "[PrivacyFilter] Blocking external search in personal mode"
            )
            return False

        if mode == "movie":
            if user_confirmed:
                log.info("[PrivacyFilter] User confirmed external search")
                return True
            else:
                log.info(
                    "[PrivacyFilter] External search needs confirmation"
                )
                return False

        # mode == "all"
        return True


# Global instance
PRIVACY_FILTER = PrivacyFilter()
