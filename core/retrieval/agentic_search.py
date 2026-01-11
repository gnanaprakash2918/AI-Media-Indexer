"""Agentic Search with LLM Query Expansion and Scene-Level Search.

Uses LLM to expand queries intelligently at search time,
NOT hardcoded synonyms during ingestion. This enables:
- "South Indian breakfast" → [idli, dosa, sambar entities]
- "Prakash bowling" → identity entity + action entity
- Complex multi-entity queries with unlimited attributes
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


# =============================================================================
# DYNAMIC QUERY EXPANSION PROMPT (NO HARDCODED CATEGORIES)
# =============================================================================

DYNAMIC_QUERY_PROMPT = """You are extracting entities from a video search query.
Extract EVERY SINGLE entity, attribute, relationship, and detail.
Do NOT use predefined categories - create entity types DYNAMICALLY.

Query: "{query}"

INSTRUCTIONS:
1. Extract ALL entities (people, vehicles, objects, clothing, sounds, text, emotions, actions, etc.)
2. For each entity, capture ALL attributes as key-value pairs
3. Capture ALL relationships between entities
4. Note temporal constraints (timing, sequence)
5. Note audio/dialogue requirements
6. Generate a dense scene description for semantic search

ENTITY TYPES (examples, not exhaustive - create new types as needed):
- person, vehicle, clothing, accessory, footwear, eyewear, bag
- object, food, drink, animal, plant, furniture
- action, gesture, emotion, expression
- sound, music, dialogue, speech
- text, sign, brand, logo
- location, setting, weather, time_of_day
- color, pattern, material, texture

OUTPUT FORMAT (JSON):
{{
  "entities": [
    {{
      "entity_type": "<dynamic type>",
      "name": "<entity identifier>",
      "attributes": {{<any key-value pairs>}},
      "relationships": [{{"relation": "<verb>", "target": "<other entity>"}}]
    }}
  ],
  "relationships": [{{"source": "<entity>", "relation": "<verb>", "target": "<entity>"}}],
  "scene_description": "<dense searchable text>",
  "temporal_constraints": ["<time hints>"],
  "audio_constraints": ["<audio/speech hints>"],
  "modalities": ["visual", "audio", "dialogue", "text"]
}}

EXAMPLE:
Query: "Prakash wearing blue t-shirt with John Jacobs spectacles and red shoe on left foot with green shoe on right foot playing bowling at Brunswick hitting a strike where last pin slowly falls"

Output:
{{
  "entities": [
    {{"entity_type": "person", "name": "Prakash", "attributes": {{}}, "relationships": [
      {{"relation": "wearing", "target": "blue t-shirt"}},
      {{"relation": "wearing", "target": "John Jacobs spectacles"}},
      {{"relation": "wearing on left foot", "target": "red shoe"}},
      {{"relation": "wearing on right foot", "target": "green shoe"}},
      {{"relation": "playing", "target": "bowling"}},
      {{"relation": "achieving", "target": "strike"}}
    ]}},
    {{"entity_type": "clothing", "name": "t-shirt", "attributes": {{"color": "blue", "body_part": "upper body"}}}},
    {{"entity_type": "eyewear", "name": "spectacles", "attributes": {{"brand": "John Jacobs"}}}},
    {{"entity_type": "footwear", "name": "shoe", "attributes": {{"color": "red", "body_part": "left foot"}}}},
    {{"entity_type": "footwear", "name": "shoe", "attributes": {{"color": "green", "body_part": "right foot"}}}},
    {{"entity_type": "activity", "name": "bowling", "attributes": {{}}}},
    {{"entity_type": "action_result", "name": "strike", "attributes": {{}}}},
    {{"entity_type": "object", "name": "pin", "attributes": {{"state": "falling", "sequence": "last"}}}},
    {{"entity_type": "location", "name": "Brunswick", "attributes": {{}}}}
  ],
  "temporal_constraints": ["slowly", "last"],
  "audio_constraints": [],
  "scene_description": "Prakash in blue t-shirt John Jacobs spectacles red and green shoes bowling at Brunswick hitting strike with pins falling slowly",
  "modalities": ["visual"]
}}

Extract EVERYTHING. Return ONLY valid JSON."""

# Legacy prompt for backwards compatibility
QUERY_EXPANSION_PROMPT = DYNAMIC_QUERY_PROMPT


class SearchAgent:
    """Agentic search with LLM-powered query expansion and scene-level search.

    Supports complex paragraph-length queries with multiple constraints:
    - Identity (face clusters)
    - Clothing color and type
    - Accessories
    - Location
    - Actions and outcomes
    - Visible text/brands
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
            query: Natural language search query (can be paragraph-length).

        Returns:
            ParsedQuery with expanded keywords and structured filters.
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
            return ParsedQuery(visual_keywords=[query])

    def _resolve_identity(self, person_name: str) -> int | None:
        """Resolve person name to cluster ID via HITL database.
        
        Uses exact match first, then fuzzy matching for:
        - Typos ("prakash" → "Prakash")
        - Partial names ("Gnana" → "Gnana Prakash")
        - Nickname variations

        Args:
            person_name: Name to look up.

        Returns:
            Cluster ID if found, None otherwise.
        """
        try:
            # 1. Try exact match first (fastest)
            cluster_id = self.db.get_cluster_id_by_name(person_name)
            if cluster_id is not None:
                return cluster_id

            # 2. Fallback to fuzzy match (handles typos, partial names)
            cluster_id = self.db.fuzzy_get_cluster_id_by_name(person_name)
            if cluster_id:
                log(f"[Search] Fuzzy matched '{person_name}' → cluster {cluster_id}")
            return cluster_id
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

    @observe("search_scenes_agentic")
    async def search_scenes(
        self,
        query: str,
        limit: int = 20,
        use_expansion: bool = True,
        video_path: str | None = None,
    ) -> dict[str, Any]:
        """Search scenes with comprehensive filtering for complex queries.

        This is the production-grade search method using scene-level storage.
        Supports complex queries like:
        "Prakash wearing blue shirt with spectacles bowling at Brunswick hitting a strike"

        Args:
            query: Natural language search query (can be paragraph-length).
            limit: Maximum results.
            use_expansion: Whether to use LLM query expansion.
            video_path: Optional filter by specific video.

        Returns:
            Dict with results, parsed query, and metadata.
        """
        log(f"[Search] Scene search: '{query[:100]}...'")

        # 1. Parse and expand query
        if use_expansion:
            parsed = await self.parse_query(query)
        else:
            parsed = ParsedQuery(visual_keywords=[query])

        # 2. Resolve identity if person name found
        cluster_id: int | None = None
        face_ids: list[str] = []
        resolved_name: str | None = None

        if parsed.person_name:
            cluster_id = self._resolve_identity(parsed.person_name)
            if cluster_id is not None:
                face_ids = self._get_face_ids_for_cluster(cluster_id)
                resolved_name = parsed.person_name
                log(f"[Search] Resolved '{parsed.person_name}' → cluster {cluster_id}")

        # 3. Build search query from expanded keywords
        search_text = parsed.to_search_text()
        log(f"[Search] Expanded search text: '{search_text}'")

        # 4. Execute scene search with comprehensive filters
        try:
            results = self.db.search_scenes(
                query=search_text,
                limit=limit,
                person_name=resolved_name,
                face_cluster_ids=[cluster_id] if cluster_id else None,
                clothing_color=parsed.clothing_color,
                clothing_type=parsed.clothing_type,
                accessories=parsed.accessories if parsed.accessories else None,
                location=parsed.location,
                visible_text=parsed.text_to_find if parsed.text_to_find else None,
                action_keywords=parsed.action_keywords if parsed.action_keywords else None,
                video_path=video_path,
                search_mode="hybrid",
            )
            log(f"[Search] Found {len(results)} scene results")
        except Exception as e:
            log(f"[Search] Scene search failed: {e}, falling back to frame search")
            results = await self._fallback_frame_search(parsed, search_text, limit)

        return {
            "query": query,
            "parsed": parsed.model_dump(),
            "resolved_identity": resolved_name,
            "face_ids_matched": len(face_ids),
            "expanded_search": search_text,
            "results": results,
            "result_count": len(results),
            "search_type": "scene",
        }

    @observe("search_agentic")
    async def search(
        self,
        query: str,
        limit: int = 20,
        use_expansion: bool = True,
    ) -> dict[str, Any]:
        """Perform agentic search with query expansion.

        This method searches FRAMES (legacy). For production, use search_scenes().

        Args:
            query: Natural language search query.
            limit: Maximum results to return.
            use_expansion: Whether to use LLM query expansion.

        Returns:
            Dict with results, parsed query, and metadata.
        """
        log(f"[Search] Agentic frame search: '{query}'")

        # 1. Parse and expand query
        if use_expansion:
            parsed = await self.parse_query(query)
        else:
            parsed = ParsedQuery(visual_keywords=[query])

        # 2. Resolve identity if person name found
        face_ids: list[str] = []
        resolved_name: str | None = None
        cluster_id: int | None = None

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

        # Identity filter
        if cluster_id is not None:
            filters.append(
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchAny(any=[cluster_id]),
                )
            )
            log(f"[Search] Added identity filter: cluster_id={cluster_id}")

        # Clothing filters (new for complex queries)
        if parsed.clothing_color:
            filters.append(
                models.FieldCondition(
                    key="clothing_colors",
                    match=models.MatchAny(any=[parsed.clothing_color.lower()]),
                )
            )
            log(f"[Search] Added clothing color filter: {parsed.clothing_color}")

        if parsed.clothing_type:
            filters.append(
                models.FieldCondition(
                    key="clothing_types",
                    match=models.MatchAny(any=[parsed.clothing_type.lower()]),
                )
            )
            log(f"[Search] Added clothing type filter: {parsed.clothing_type}")

        # Accessories filter
        if parsed.accessories:
            filters.append(
                models.FieldCondition(
                    key="accessories",
                    match=models.MatchAny(any=parsed.accessories),
                )
            )
            log(f"[Search] Added accessories filter: {parsed.accessories}")

        # Brand/Text filter
        if parsed.text_to_find:
            filters.append(
                models.FieldCondition(
                    key="visible_text",
                    match=models.MatchAny(any=parsed.text_to_find),
                )
            )
            log(f"[Search] Added text filter: {parsed.text_to_find}")

        # Location filter
        if parsed.location:
            filters.append(
                models.FieldCondition(
                    key="scene_location",
                    match=models.MatchText(text=parsed.location),
                )
            )
            log(f"[Search] Added location filter: {parsed.location}")

        # Entity/Object filter
        if parsed.visual_keywords:
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
                conditions: list[models.Condition] = list(filters)
                results = self.db.client.query_points(
                    collection_name=self.db.MEDIA_COLLECTION,
                    query=query_vector,
                    query_filter=models.Filter(should=conditions) if len(conditions) > 1 else models.Filter(must=conditions),
                    limit=limit,
                ).points
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
            "search_type": "frame",
        }

    async def _fallback_frame_search(
        self,
        parsed: ParsedQuery,
        search_text: str,
        limit: int,
    ) -> list[dict]:
        """Fallback to frame search if scene search fails."""
        try:
            return self.db.search_frames(query=search_text, limit=limit)
        except Exception:
            return []

    async def search_simple(self, query: str, limit: int = 20) -> list[dict]:
        """Simple search without expansion (fallback).

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of search results.
        """
        return self.db.search_frames(query=query, limit=limit)

    # =========================================================================
    # SOTA SEARCH METHODS
    # =========================================================================

    @observe("search_rerank_llm")
    async def rerank_with_llm(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """Re-rank candidates using LLM to verify ALL query constraints.
        
        This is the SOTA re-ranking stage that:
        1. Takes candidate results from vector search
        2. Uses LLM to verify each constraint from query is satisfied
        3. Scores based on number of matched constraints
        4. Provides chain-of-thought reasoning for each decision
        
        Args:
            query: Original user query.
            candidates: List of search results to re-rank.
            top_k: Number of top results to return.
        
        Returns:
            Re-ranked list with LLM verification scores and reasoning.
        """
        if not candidates:
            return []

        # Re-ranking prompt template
        RERANK_PROMPT = """You are verifying if a video segment matches a user query.

QUERY: "{query}"

SEGMENT DESCRIPTION:
{description}

SEGMENT METADATA:
- People: {face_names}
- Location: {location}
- Actions: {actions}
- Visible Text: {visible_text}

TASK: Analyze how well this segment matches the query.

For EACH constraint in the query, determine if it's satisfied:
1. Check ALL people mentioned - are they present?
2. Check ALL clothing/accessories - do they match?
3. Check ALL actions - are they occurring?
4. Check ALL locations/brands - do they match?
5. Check temporal requirements - timing correct?

Return JSON:
{{
  "match_score": 0.0-1.0,  // Overall match (0=no match, 1=perfect match)
  "constraints_checked": [
    {{"constraint": "<what to check>", "satisfied": true/false, "evidence": "<why>"}}
  ],
  "reasoning": "<chain of thought explaining the score>",
  "missing": ["<constraints not satisfied>"]
}}"""

        reranked = []

        for candidate in candidates[:top_k * 2]:  # Check more candidates than needed
            description = (
                candidate.get("description", "") or
                candidate.get("dense_caption", "") or
                candidate.get("raw_description", "")
            )

            prompt = RERANK_PROMPT.format(
                query=query,
                description=description[:500],
                face_names=candidate.get("face_names", []),
                location=candidate.get("location", "unknown"),
                actions=candidate.get("actions", []),
                visible_text=candidate.get("visible_text", []),
            )

            try:
                # Use LLM to score this candidate
                from pydantic import BaseModel, Field

                class RerankResult(BaseModel):
                    match_score: float = Field(default=0.5)
                    constraints_checked: list[dict] = Field(default_factory=list)
                    reasoning: str = Field(default="")
                    missing: list[str] = Field(default_factory=list)

                result = await self.llm.generate_structured(
                    schema=RerankResult,
                    prompt=prompt,
                    system_prompt="You are a video search result verifier. Return JSON only.",
                )

                # Merge LLM verification with candidate
                candidate["llm_score"] = result.match_score
                candidate["llm_reasoning"] = result.reasoning
                candidate["constraints_satisfied"] = [
                    c for c in result.constraints_checked if c.get("satisfied")
                ]
                candidate["constraints_missing"] = result.missing
                candidate["combined_score"] = (
                    candidate.get("score", 0) * 0.4 +  # Vector similarity
                    result.match_score * 0.6           # LLM verification
                )
                reranked.append(candidate)

            except Exception as e:
                # If LLM fails, keep original score
                log(f"[Rerank] LLM verification failed: {e}")
                candidate["combined_score"] = candidate.get("score", 0)
                reranked.append(candidate)

        # Sort by combined score
        reranked.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return reranked[:top_k]

    @observe("search_sota")
    async def sota_search(
        self,
        query: str,
        limit: int = 10,
        video_path: str | None = None,
        use_reranking: bool = True,
    ) -> dict[str, Any]:
        """SOTA search pipeline with full verification.
        
        This is the highest quality search method that:
        1. Parses query with dynamic entity extraction
        2. Retrieves candidates via multi-vector hybrid search
        3. Re-ranks with LLM constraint verification
        4. Returns explainable results with reasoning
        
        Args:
            query: Complex natural language query.
            limit: Maximum results.
            video_path: Optional filter by video.
            use_reranking: Whether to use LLM re-ranking (slower but more accurate).
        
        Returns:
            Dict with ranked results and full explainability.
        """
        log(f"[SOTA Search] Query: '{query[:100]}...'")

        # 1. Parse query with dynamic entity extraction
        parsed = await self.parse_query(query)
        search_text = parsed.to_search_text() or query
        log(f"[SOTA Search] Expanded: '{search_text[:100]}...'")

        # 2. Resolve person identities
        person_names = []
        face_ids = []

        # Try new dynamic format first
        if hasattr(parsed, 'entities') and parsed.entities:
            for entity in parsed.entities:
                if entity.entity_type.lower() == "person" and entity.name:
                    person_names.append(entity.name)
        # Fallback to legacy format
        elif parsed.person_name:
            person_names.append(parsed.person_name)

        for name in person_names:
            cluster_id = self._resolve_identity(name)
            if cluster_id:
                ids = self._get_face_ids_for_cluster(cluster_id)
                face_ids.extend(ids)
                log(f"[SOTA Search] Resolved '{name}' → {len(ids)} faces")

        # 3. Retrieve candidates via multi-vector search
        try:
            candidates = self.db.explainable_search(
                query_text=search_text,
                parsed_query=parsed,
                limit=limit * 3 if use_reranking else limit,
                score_threshold=0.25,
            )
            log(f"[SOTA Search] Retrieved {len(candidates)} candidates")
        except Exception as e:
            log(f"[SOTA Search] Search failed: {e}")
            candidates = []

        # 4. Re-rank with LLM verification (optional but recommended)
        if use_reranking and candidates:
            try:
                candidates = await self.rerank_with_llm(query, candidates, top_k=limit)
                log(f"[SOTA Search] Re-ranked to {len(candidates)} results")
            except Exception as e:
                log(f"[SOTA Search] Re-ranking failed: {e}, using raw results")

        # 5. Build response with full explainability
        return {
            "query": query,
            "parsed": parsed.model_dump() if hasattr(parsed, 'model_dump') else {},
            "search_text": search_text,
            "person_names_resolved": person_names,
            "face_ids_matched": len(face_ids),
            "results": candidates[:limit],
            "result_count": len(candidates[:limit]),
            "search_type": "sota",
            "reranking_used": use_reranking,
        }
