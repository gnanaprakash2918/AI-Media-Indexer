"""External Metadata Engine - TMDB Integration and Cast Mapping.

Fetches movie/TV metadata from TMDB API to:
1. Auto-create placeholder Face Clusters with cast names
2. Query external movie databases
3. Speed up HITL identity tagging
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

TMDB_API_BASE = "https://api.themoviedb.org/3"


@dataclass
class CastMember:
    name: str
    character: str
    profile_path: Optional[str] = None
    order: int = 0
    tmdb_id: int = 0


@dataclass
class MediaMetadata:
    """Dataclass metadata for media content (movies/TV)."""
    title: str
    year: Optional[int] = None
    overview: str = ""
    poster_path: Optional[str] = None
    tmdb_id: int = 0
    media_type: str = "movie"
    cast: list[CastMember] | None = None

    def __post_init__(self):
        if self.cast is None:
            self.cast = []


class MetadataEngine:
    """Fetches metadata from TMDB and manages cast-to-face mapping."""

    def __init__(
        self,
        tmdb_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        tmdb_key: Optional[str] = None,
        **kwargs,
    ):
        self.api_key = tmdb_api_key or api_key or tmdb_key or os.getenv("TMDB_API_KEY")
        self._client: Optional[httpx.AsyncClient] = None
        self.enabled = bool(self.api_key)

        if not self.enabled:
            logger.warning(
                "[MetadataEngine] TMDB_API_KEY not set. TMDB metadata disabled."
            )

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    @property
    def is_available(self) -> bool:
        """Check if metadata engine is configured and enabled."""
        return self.enabled

    async def search_movie(
        self, query: str, year: Optional[int] = None
    ) -> list[MediaMetadata]:
        """Search TMDB for movies matching query."""
        if not self.api_key:
            logger.warning("TMDB_API_KEY not set. Cannot search movies.")
            return []

        params = {
            "api_key": self.api_key,
            "query": query,
            "include_adult": "false",
        }
        if year:
            params["year"] = str(year)

        try:
            resp = await self.client.get(f"{TMDB_API_BASE}/search/movie", params=params)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", [])[:5]:
                release_date = item.get("release_date", "")
                year = (
                    int(release_date[:4])
                    if release_date and len(release_date) >= 4
                    else None
                )
                results.append(
                    MediaMetadata(
                        title=item.get("title", ""),
                        year=year,
                        overview=item.get("overview", ""),
                        poster_path=item.get("poster_path"),
                        tmdb_id=item.get("id", 0),
                        media_type="movie",
                    )
                )
            return results
        except Exception as e:
            logger.error(f"TMDB search failed: {e}")
            return []

    async def get_cast(
        self, tmdb_id: int, media_type: str = "movie"
    ) -> list[CastMember]:
        """Get cast list for a movie/TV show."""
        if not self.api_key:
            return []

        try:
            resp = await self.client.get(
                f"{TMDB_API_BASE}/{media_type}/{tmdb_id}/credits",
                params={"api_key": self.api_key},
            )
            resp.raise_for_status()
            data = resp.json()

            cast_list = []
            for member in data.get("cast", [])[:20]:
                cast_list.append(
                    CastMember(
                        name=member.get("name", ""),
                        character=member.get("character", ""),
                        profile_path=member.get("profile_path"),
                        order=member.get("order", 0),
                        tmdb_id=member.get("id", 0),
                    )
                )
            return cast_list
        except Exception as e:
            logger.error(f"TMDB cast fetch failed: {e}")
            return []

    async def get_movie_with_cast(
        self, query: str, year: Optional[int] = None
    ) -> Optional[MediaMetadata]:
        """Search for movie and fetch its cast in one call."""
        results = await self.search_movie(query, year)
        if not results:
            return None

        movie = results[0]
        movie.cast = await self.get_cast(movie.tmdb_id, "movie")
        return movie

    async def close(self):
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def enrich_video(
        self, video_id: str | None, path: Path | str
    ) -> Optional[MediaMetadata]:
        """Enrich video metadata using filename parsing and TMDB (if enabled)."""
        if not self.enabled:
            return None

        try:
            filename = Path(path).name

            # Simple heuristic: Look for (YYYY) pattern
            year_match = re.search(r"\((\d{4})\)", filename)
            year = int(year_match.group(1)) if year_match else None

            # Clean title
            title = filename
            title = re.sub(r"\.\w+$", "", title)
            title = re.sub(r"\((\d{4})\).*", "", title)
            title = re.sub(r"\s+", " ", title).strip()

            # Search TMDB if enabled and looks like a movie
            if self.enabled and year:
                found = await self.get_movie_with_cast(title, year)
                if found:
                    logger.info(f"[Metadata] TMDB Match: {found.title} ({found.year})")
                    return found

            return MediaMetadata(
                title=title, year=year, media_type="movie" if year else "video"
            )
        except Exception as e:
            logger.warning(f"[Metadata] Enrich failed: {e}")
            return None

    async def identify(
        self, path: Path | str, user_hint: Optional[Any] = None
    ) -> Optional[MediaMetadata]:
        """Alias for enrich_video to maintain compatibility."""
        return await self.enrich_video(None, path)


def fuzzy_match_name(name1: str, name2: str, threshold: float = 0.8) -> bool:
    """Simple fuzzy name matching using token overlap."""

    def normalize(s: str) -> set[str]:
        s = re.sub(r"[^\w\s]", "", s.lower())
        return set(s.split())

    tokens1 = normalize(name1)
    tokens2 = normalize(name2)

    if not tokens1 or not tokens2:
        return False

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    jaccard = len(intersection) / len(union)
    return jaccard >= threshold


def match_cast_to_clusters(
    cast_list: list[CastMember],
    face_clusters: list[dict],
) -> dict[int, str]:
    """Match TMDB cast to existing face clusters using fuzzy matching.

    Returns mapping of cluster_id -> cast name (unverified).
    """
    matches: dict[int, str] = {}

    for cluster in face_clusters:
        cluster_id = cluster.get("cluster_id")
        cluster_name = (cluster.get("name") or cluster.get("label") or "").strip()

        if not cluster_name or cluster_name.lower() in ("unknown", "unnamed"):
            continue

        for cast in cast_list:
            if fuzzy_match_name(cast.name, cluster_name):
                matches[cluster_id] = cast.name
                logger.info(
                    f"Matched cluster {cluster_id} ({cluster_name}) -> {cast.name}"
                )
                break

    return matches


def create_placeholder_clusters_from_cast(cast_list: list[CastMember]) -> list[dict]:
    """Create placeholder face cluster entries from TMDB cast.

    Uses String UUIDs with 'tmdb_' prefix instead of negative integers
    to prevent the 'Voice Cluster #-1' bug and enable proper HITL naming.
    """
    import uuid

    placeholders = []
    for i, cast in enumerate(cast_list[:10]):
        # Generate a unique cluster ID that won't collide with real clusters
        # Format: "tmdb_<tmdb_person_id>_<uuid_suffix>" for traceability
        cluster_uuid = f"tmdb_{cast.tmdb_id}_{uuid.uuid4().hex[:8]}"
        placeholders.append(
            {
                "cluster_id": cluster_uuid,  # String UUID instead of negative int
                "name": cast.name,
                "character": cast.character,
                "verified": False,
                "source": "tmdb_placeholder",  # Mark as unverified placeholder
                "tmdb_id": cast.tmdb_id,
                "profile_url": f"https://image.tmdb.org/t/p/w185{cast.profile_path}"
                if cast.profile_path
                else None,
            }
        )
    return placeholders


_engine: Optional[MetadataEngine] = None


def get_metadata_engine() -> MetadataEngine:
    global _engine
    if _engine is None:
        _engine = MetadataEngine()
    return _engine
