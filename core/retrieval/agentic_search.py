"""Agentic Search with LLM Query Expansion.

Uses LLM to expand queries intelligently at search time,
NOT hardcoded synonyms during ingestion. This enables:
- "South Indian breakfast" → ["idli", "dosa", "sambar"]
- "Prakash bowling" → identity filter + action search
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.knowledge.schemas import ParsedQuery
from core.utils.logger import log
from core.utils.observe import observe
from llm.factory import LLMFactory

if TYPE_CHECKING:
    from core.storage.db import VectorDB
    from llm.interface import LLMInterface


QUERY_EXPANSION_PROMPT = """You are a video search query analyzer. Parse and EXPAND the user's query.

User Query: "{query}"

INSTRUCTIONS:
1. Extract person name if mentioned (proper nouns only, not "man" or "woman")
2. EXPAND visual keywords (e.g., "South Indian breakfast" → ["idli", "dosa", "vada", "sambar"])
3. EXPAND cultural terms (e.g., "Indian wedding" → ["saree", "mandap", "haldi", "mehendi"])
4. Identify action keywords (bowling, eating, running, dancing)
5. Extract specific text/brands to search (Nike, LensKart, Brunswick)
6. Note temporal hints (slowly, quickly, at the end)

EXAMPLES:
- "Prakash eating south indian food" → person: "Prakash", visual: ["idli", "dosa", "sambar", "rice", "curry"]
- "Someone bowling a strike at Brunswick" → action: ["bowling", "strike"], text: ["Brunswick"]
- "The pin fell slowly" → action: ["pin falling"], temporal: ["slowly"]

Return structured JSON matching ParsedQuery schema."""


class SearchAgent:
    """Agentic search with LLM-powered query expansion.

    The key insight: expand synonyms at SEARCH time, not ingestion time.
    This way "South Indian breakfast" finds frames indexed with "idli".
    """

    def __init__(self, db: VectorDB, llm: LLMInterface | None = None):
        """Initialize search agent.

        Args:
            db: Vector database for search.
            llm: LLM for query expansion.
        """
        self.db = db
        self.llm = llm or LLMFactory.create_llm()

    @observe("search_parse_query")
    async def parse_query(self, query: str) -> ParsedQuery:
        """Parse and expand user query using LLM.

        Args:
            query: Natural language search query.

        Returns:
            ParsedQuery with expanded keywords.
        """
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)

        try:
            parsed = await self.llm.generate_structured(
                schema=ParsedQuery,
                prompt=prompt,
                system_prompt="You are a search query parser. Return JSON only.",
            )
            log(f"[Search] Parsed query: {parsed.model_dump()}")
            return parsed

        except Exception as e:
            log(f"[Search] Query parsing failed: {e}, using raw query")
            # Fallback: treat entire query as visual keywords
            return ParsedQuery(visual_keywords=[query])

    def _resolve_identity(self, person_name: str) -> int | None:
        """Resolve person name to cluster ID via HITL database.

        Args:
            person_name: Name to look up.

        Returns:
            Cluster ID if found, None otherwise.
        """
        try:
            # Search faces collection for this name
            return self.db.get_cluster_id_by_name(person_name)
        except Exception:
            return None

    def _get_face_ids_for_cluster(self, cluster_id: int) -> list[str]:
        """Get all face point IDs belonging to a cluster.

        Args:
            cluster_id: The cluster to look up.

        Returns:
            List of face IDs in that cluster.
        """
        try:
            return self.db.get_face_ids_by_cluster(cluster_id)
        except Exception:
            return []

    @observe("search_agentic")
    async def search(
        self,
        query: str,
        limit: int = 20,
        use_expansion: bool = True,
    ) -> dict[str, Any]:
        """Perform agentic search with query expansion.

        Args:
            query: Natural language search query.
            limit: Maximum results to return.
            use_expansion: Whether to use LLM query expansion.

        Returns:
            Dict with results, parsed query, and metadata.
        """
        log(f"[Search] Agentic search: '{query}'")

        # 1. Parse and expand query
        if use_expansion:
            parsed = await self.parse_query(query)
        else:
            parsed = ParsedQuery(visual_keywords=[query])

        # 2. Resolve identity if person name found
        face_ids: list[str] = []
        resolved_name: str | None = None

        if parsed.person_name:
            cluster_id = self._resolve_identity(parsed.person_name)
            if cluster_id is not None:
                face_ids = self._get_face_ids_for_cluster(cluster_id)
                resolved_name = parsed.person_name
                log(f"[Search] Resolved '{parsed.person_name}' → cluster {cluster_id} ({len(face_ids)} faces)")

        # 3. Build search query from expanded keywords
        search_text = parsed.to_search_text()
        log(f"[Search] Expanded search text: '{search_text}'")

        # 4. Build HYBRID FILTERS for precise matching
        from qdrant_client.http import models

        filters: list[models.FieldCondition] = []

        # Identity filter: Use face_cluster_ids (from proper clustering)
        if parsed.person_name:
            cluster_id = self._resolve_identity(parsed.person_name)
            if cluster_id is not None:
                filters.append(
                    models.FieldCondition(
                        key="face_cluster_ids",
                        match=models.MatchAny(any=[cluster_id]),
                    )
                )
                log(f"[Search] Added identity filter: cluster_id={cluster_id}")

        # Brand/Text filter: Match visible_text (OCR results)
        if parsed.text_to_find:
            filters.append(
                models.FieldCondition(
                    key="visible_text",
                    match=models.MatchAny(any=parsed.text_to_find),
                )
            )
            log(f"[Search] Added text filter: {parsed.text_to_find}")

        # Entity/Object filter: Match specific items (Nike, Idly, etc.)
        if parsed.visual_keywords:
            # Only filter by entity names if they're specific (not generic)
            specific_entities = [k for k in parsed.visual_keywords if len(k) > 3]
            if specific_entities:
                filters.append(
                    models.FieldCondition(
                        key="entity_names",
                        match=models.MatchAny(any=specific_entities),
                    )
                )
                log(f"[Search] Added entity filter: {specific_entities}")

        # 5. Execute HYBRID search (Vector + Filters)
        try:
            query_vector = self.db.encode_texts(search_text or "scene activity", is_query=True)[0]
            
            if filters:
                # Cast filters to List[Condition] for Qdrant
                conditions: list[models.Condition] = list(filters)
                
                # Apply filters via Qdrant
                results = self.db.client.query_points(
                    collection_name=self.db.MEDIA_COLLECTION,
                    query=query_vector,
                    query_filter=models.Filter(should=conditions) if len(conditions) > 1 else models.Filter(must=conditions),
                    limit=limit,
                ).points
                # Format results
                results = [
                    {
                        "score": hit.score,
                        "id": str(hit.id),
                        **(hit.payload or {}),
                    }
                    for hit in results
                ]
            else:
                results = self.db.search_frames(
                    query=search_text,
                    limit=limit,
                )
        except Exception as e:
            log(f"[Search] Hybrid search failed: {e}, falling back to simple")
            results = self.db.search_frames(
                query=search_text,
                limit=limit,
            )

        return {
            "query": query,
            "parsed": parsed.model_dump(),
            "resolved_identity": resolved_name,
            "face_ids_matched": len(face_ids),
            "expanded_search": search_text,
            "results": results,
            "result_count": len(results),
        }

    async def search_simple(self, query: str, limit: int = 20) -> list[dict]:
        """Simple search without expansion (fallback).

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of search results.
        """
        return self.db.search_frames(query=query, limit=limit)
