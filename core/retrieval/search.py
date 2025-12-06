from typing import Any

from core.storage.db import VectorDB


class SearchEngine:
    def __init__(self, db: VectorDB):
        self.db = db

    def search(self, query: str, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Performs a multi-modal search across visual and dialogue collections.

        Args:
            query: The search string (e.g., "red car" or "hello").
            limit: The maximum number of results to return per category.

        Returns:
            A dictionary with keys 'visual_matches' and 'dialogue_matches'.
        """
        print(f"[Search] Querying: '{query}'...")

        visual_results = self.db.search_frames(query, limit=limit)
        dialogue_results = self.db.search_media(query, limit=limit)

        results = {
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
