"""Agentic Search Engine with LLM Query Expansion.

Parses queries to expand "South Indian breakfast" → "idly, dosa, sambar"
and resolves person names to face cluster IDs for identity filtering.
"""

import json
from typing import Any

from core.storage.db import VectorDB
from core.utils.logger import log
from core.utils.observe import observe
from llm.factory import LLMFactory


class SearchEngine:
    """Search engine with agentic query expansion and identity filtering."""

    def __init__(self, db: VectorDB) -> None:
        self.db = db
        self._llm = None

    @property
    def llm(self):
        """Lazy-load LLM to avoid init delays."""
        if self._llm is None:
            self._llm = LLMFactory.create_llm()
        return self._llm

    @observe("search_agentic")
    async def search_agentic(
        self,
        user_query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Intelligent search with query expansion and identity filtering.

        Steps:
        1. Parse query to extract person names, visual keywords (expand synonyms)
        2. Resolve person name to face cluster ID
        3. Filter search by cluster IDs if person found
        4. Vector search on expanded keywords

        Args:
            user_query: Natural language query like "Prakash eating idly"
            limit: Max results to return

        Returns:
            Dict with visual_matches and metadata
        """
        log(f"[Search] Agentic search: '{user_query}'")

        # 1. Parse and expand query
        query_data = await self._parse_query(user_query)
        log(f"[Search] Parsed: {query_data}")

        # 2. Resolve identity
        cluster_ids: list[int] = []
        person_name = query_data.get("person_name")
        if person_name:
            cid = self.db.get_cluster_id_by_name(person_name)
            if cid:
                cluster_ids.append(cid)
                log(f"[Search] Resolved '{person_name}' → cluster {cid}")

        # 3. Build search query from expanded keywords
        visual_keywords = query_data.get("visual_keywords", user_query)
        text_keywords = query_data.get("text_in_scene", "")
        search_text = f"{visual_keywords} {text_keywords}".strip()
        log(f"[Search] Expanded search: '{search_text}'")

        # 4. Execute search with optional identity filter
        query_vector = self.db.encode_texts(search_text, is_query=True)[0]
        results = self.db.search_frames_filtered(
            query_vector=query_vector,
            face_cluster_ids=cluster_ids if cluster_ids else None,
            limit=limit,
        )

        # Format results
        formatted = []
        for hit in results:
            formatted.append(
                {
                    "score": f"{hit.get('score', 0):.2f}",
                    "time": f"{hit.get('timestamp', 0):.2f}s",
                    "file": hit.get("video_path"),
                    "content": hit.get("action"),
                    "details": hit.get("structured_data"),
                }
            )

        return {
            "query": user_query,
            "parsed": query_data,
            "resolved_identity": person_name if cluster_ids else None,
            "expanded_search": search_text,
            "visual_matches": formatted,
        }

    async def _parse_query(self, query: str) -> dict[str, Any]:
        """Parse query and expand visual keywords using LLM."""
        prompt = f"""Parse this video search query: "{query}"

Return JSON with:
{{
    "person_name": "Name of person/character if mentioned, or null",
    "visual_keywords": "Expanded visual keywords with specific items",
    "text_in_scene": "Any specific text/brand mentioned (e.g., LensKart, Tesla) or empty"
}}

UNIVERSAL EXPANSION RULES - Expand generic terms to specific items:
- "breakfast" → "idly dosa sambar eggs toast cereal pancakes"
- "South Indian food" → "idly dosa sambar rasam vada pongal biryani"
- "Japanese food" → "sushi ramen tempura miso onigiri"
- "car" → "sedan SUV Tesla BMW Toyota sports car"
- "weapon" → "katana sword knife gun rifle AK-47 lightsaber"
- "phone" → "iPhone Samsung Galaxy smartphone mobile"
- "shoes" → "sneakers Nike Adidas Jordan boots heels"
- "eating" → "eating dipping biting chewing swallowing"
- "fighting" → "punching kicking slashing blocking dodging"
- "driving" → "driving steering accelerating drifting braking"

Return ONLY JSON, no explanation."""

        try:
            raw = await self.llm.generate(prompt)
            # Extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            log(f"[Search] Query parsing failed: {e}")

        # Fallback
        return {"person_name": None, "visual_keywords": query}

    def search(self, query: str, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Synchronous search (backward compatible)."""
        frame_results = self.db.search_frames(query, limit=limit)
        dialogue_results = self.db.search_media(query, limit=limit)

        return {
            "visual_matches": frame_results,
            "dialogue_matches": dialogue_results,
        }
