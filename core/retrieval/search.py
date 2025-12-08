"""High-level multimodal search over indexed media.

This module defines the :class:`SearchEngine`, a thin facade around
:class:`VectorDB` that performs query-time fusion over both visual
(frame-based) and dialogue (subtitle/transcript) collections.

The engine returns results in a simple, JSON-friendly structure:
separate lists for visual and dialogue matches, each with a score,
timestamp, file path, and content snippet.
"""

from typing import Any

from core.storage.db import VectorDB


class SearchEngine:
    """High-level search entrypoint for multimedia queries.

    This class encapsulates the logic for querying both frame vectors and
    dialogue segments from the underlying :class:`VectorDB` instance and
    combines them into a single structured response.
    """

    def __init__(self, db: VectorDB) -> None:
        """Initialize the search engine.

        Args:
            db: A configured :class:`VectorDB` instance used to perform
                similarity search over both frames and media segments.
        """
        self.db = db

    def search(self, query: str, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Perform a multimodal search across visual and dialogue collections.

        Args:
            query: Natural language search string (for example, ``"red car"``,
                ``"argument in a kitchen"``, or ``"hello"``).
            limit: Maximum number of results to return per category.

        Returns:
            A dictionary with two keys:

            * ``"visual_matches"``: List of frame-based matches, each with
              ``"score"``, ``"time"``, ``"file"``, and ``"content"``.
            * ``"dialogue_matches"``: List of dialogue-based matches with the
              same keys.
        """
        print(f"[Search] Querying: '{query}'...")

        visual_results = self.db.search_frames(query, limit=limit)
        dialogue_results = self.db.search_media(query, limit=limit)

        results: dict[str, list[dict[str, Any]]] = {
            "visual_matches": [],
            "dialogue_matches": [],
        }

        for hit in visual_results:
            match = {
                "score": f"{hit.get('score', 0):.2f}",
                "time": f"{hit.get('timestamp', 0):.2f}s",
                "file": hit.get("video_path"),
                "content": hit.get("action") or "Visual scene",
            }
            results["visual_matches"].append(match)

        for hit in dialogue_results:
            match = {
                "score": f"{hit.get('score', 0):.2f}",
                "time": f"{hit.get('start', 0):.2f}s",
                "file": hit.get("video_path"),
                "content": hit.get("text") or "Dialogue",
            }
            results["dialogue_matches"].append(match)

        return results
