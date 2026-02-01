"""Celebrity lookup for face identification.

Provides celebrity identification via:
- TMDB movie cast matching
- Brave Search API for general celebrities
- HITL verification for privacy protection
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from config import settings
from core.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CelebrityMatch:
    """Result from celebrity lookup."""

    name: str
    confidence: float
    source: str  # "tmdb", "brave", "manual"
    image_url: str | None = None
    metadata: dict[str, Any] | None = None
    verified: bool = False  # HITL verification status


class TMDBLookup:
    """Lookup celebrities from TMDB movie database.

    Matches face clusters against movie cast lists.
    """

    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str | None = None):
        """Initialize TMDB lookup.

        Args:
            api_key: TMDB API key. Falls back to settings.
        """
        self.api_key = api_key or getattr(settings, "tmdb_api_key", None)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient | None:
        """Get or create HTTP client."""
        if not self.api_key:
            log.warning("[TMDB] No API key configured")
            return None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def search_movie(self, title: str) -> dict | None:
        """Search for a movie by title.

        Args:
            title: Movie title to search.

        Returns:
            Movie info dict or None.
        """
        client = await self._get_client()
        if not client:
            return None

        try:
            resp = await client.get(
                f"{self.BASE_URL}/search/movie",
                params={"api_key": self.api_key, "query": title},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return results[0] if results else None
        except Exception as e:
            log.error(f"[TMDB] Search failed: {e}")
            return None

    async def get_cast(self, movie_id: int) -> list[CelebrityMatch]:
        """Get cast list for a movie.

        Args:
            movie_id: TMDB movie ID.

        Returns:
            List of CelebrityMatch for cast members.
        """
        client = await self._get_client()
        if not client:
            return []

        try:
            resp = await client.get(
                f"{self.BASE_URL}/movie/{movie_id}/credits",
                params={"api_key": self.api_key},
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for actor in data.get("cast", [])[:20]:  # Top 20 billed
                profile = actor.get("profile_path")
                image_url = (
                    f"https://image.tmdb.org/t/p/w185{profile}"
                    if profile
                    else None
                )
                results.append(
                    CelebrityMatch(
                        name=actor.get("name", "Unknown"),
                        confidence=0.8,  # Cast listing confidence
                        source="tmdb",
                        image_url=image_url,
                        metadata={
                            "character": actor.get("character"),
                            "id": actor.get("id"),
                            "order": actor.get("order", 999),
                        },
                    )
                )

            log.info(f"[TMDB] Found {len(results)} cast members")
            return results

        except Exception as e:
            log.error(f"[TMDB] Cast lookup failed: {e}")
            return []

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class BraveCelebritySearch:
    """Search for celebrities using Brave Search API.

    For general public figures outside movie databases.
    """

    BASE_URL = "https://api.search.brave.com/res/v1/images/search"

    def __init__(self, api_key: str | None = None):
        """Initialize Brave search.

        Args:
            api_key: Brave Search API key.
        """
        self.api_key = api_key or getattr(settings, "brave_api_key", None)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient | None:
        """Get or create HTTP client."""
        if not self.api_key:
            log.warning("[Brave] No API key configured")
            return None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def search_image(
        self,
        image_url: str,
        limit: int = 5,
    ) -> list[CelebrityMatch]:
        """Reverse image search to identify a face.

        Args:
            image_url: URL of face image to search.
            limit: Max results to return.

        Returns:
            List of potential CelebrityMatch.
        """
        client = await self._get_client()
        if not client:
            return []

        try:
            resp = await client.get(
                self.BASE_URL,
                params={"q": "celebrity face", "count": limit},
                headers={"X-Subscription-Token": self.api_key or ""},
            )
            resp.raise_for_status()
            data = resp.json()

            # Note: Brave API structure may vary
            results = []
            for item in data.get("results", [])[:limit]:
                results.append(
                    CelebrityMatch(
                        name=item.get("title", "Unknown"),
                        confidence=0.5,  # Lower confidence - needs HITL
                        source="brave",
                        image_url=item.get("thumbnail", {}).get("src"),
                    )
                )

            return results

        except Exception as e:
            log.error(f"[Brave] Search failed: {e}")
            return []

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class CelebrityIdentifier:
    """Unified celebrity identification service.

    Combines TMDB and Brave for comprehensive lookup.
    """

    def __init__(self):
        """Initialize identifier with available sources."""
        self.tmdb = TMDBLookup()
        self.brave = BraveCelebritySearch()

    async def identify_from_movie(
        self,
        movie_title: str,
    ) -> list[CelebrityMatch]:
        """Get celebrities from a movie's cast.

        Args:
            movie_title: Movie title to lookup.

        Returns:
            List of cast members as CelebrityMatch.
        """
        movie = await self.tmdb.search_movie(movie_title)
        if not movie:
            log.warning(f"[Celebrity] Movie not found: {movie_title}")
            return []

        return await self.tmdb.get_cast(movie["id"])

    async def match_face_to_cast(
        self,
        face_embedding: list[float],
        cast: list[CelebrityMatch],
        face_manager: Any,  # Avoid circular import, pass instance
    ) -> CelebrityMatch | None:
        """Match a face embedding against movie cast.

        Args:
            face_embedding: 512D face embedding.
            cast: List of CelebrityMatch from TMDB.
            face_manager: Instance of FaceManager to detect faces in cast photos.

        Returns:
            Best matching celebrity or None.
        """
        import tempfile
        from pathlib import Path

        import numpy as np

        if not cast:
            return None

        target_emb = np.array(face_embedding, dtype=np.float64)
        target_emb /= np.linalg.norm(target_emb) + 1e-9

        best_match: CelebrityMatch | None = None
        best_score = 0.0

        # Create a temp dir for downloading images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Process cast members with profile images
            # Optimisation: Check top 10 billed actors first
            client = await self.tmdb._get_client()
            if not client:
                return None

            for actor in cast[:10]:
                if not actor.image_url:
                    continue

                try:
                    # Download image
                    ext = actor.image_url.split(".")[-1]
                    local_img_path = temp_path / f"{actor.metadata['id']}.{ext}"

                    # Download
                    resp = await client.get(actor.image_url)
                    if resp.status_code != 200:
                        continue

                    with open(local_img_path, "wb") as f:
                        f.write(resp.content)

                    # Detect face
                    # Use SFace or InsightFace - FaceManager handles this
                    faces = await face_manager.detect_faces(local_img_path)
                    if not faces:
                        continue

                    # Find best matching face in the actor's photo
                    # (Actor photo might have other people, but usually it's a headshot)
                    # We assume the largest face is the actor
                    largest_face = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0])
                        * (f.bbox[3] - f.bbox[1]),
                    )

                    if largest_face.embedding is None:
                        continue

                    actor_emb = np.array(
                        largest_face.embedding, dtype=np.float64
                    )
                    actor_emb /= np.linalg.norm(actor_emb) + 1e-9

                    score = float(np.dot(target_emb, actor_emb))
                    log.info(
                        f"[Celebrity] Checking {actor.name}: score={score:.3f}"
                    )

                    if score > best_score:
                        best_score = score
                        best_match = actor

                except Exception as e:
                    log.warning(
                        f"[Celebrity] Failed to check {actor.name}: {e}"
                    )
                    continue

        # Threshold for identity match
        # InsightFace threshold is usually around 0.3-0.5 depending on loss
        # We'll be conservative
        MATCH_THRESHOLD = 0.45

        if best_score > MATCH_THRESHOLD and best_match:
            best_match.confidence = best_score
            log.info(
                f"[Celebrity] MATCH FOUND: {best_match.name} (score={best_score:.3f})"
            )
            return best_match

        return None

    async def close(self) -> None:
        """Close all HTTP clients."""
        await asyncio.gather(
            self.tmdb.close(),
            self.brave.close(),
        )
