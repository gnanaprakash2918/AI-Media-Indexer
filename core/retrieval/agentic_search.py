"""Agentic Search with LLM Query Expansion and Scene-Level Search.

Uses LLM to expand queries intelligently at search time,
NOT hardcoded synonyms during ingestion. Prompts loaded from external files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.knowledge.schemas import ParsedQuery
from core.utils.logger import log
from core.utils.observe import observe
from core.utils.prompt_loader import load_prompt
from llm.factory import LLMFactory

if TYPE_CHECKING:
    from core.storage.db import VectorDB
    from llm.interface import LLMInterface


# Load prompts from external files - NO HARDCODING
DYNAMIC_QUERY_PROMPT = load_prompt("dynamic_query")
QUERY_EXPANSION_PROMPT = DYNAMIC_QUERY_PROMPT  # Legacy alias


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

    def __init__(
        self,
        db: "VectorDB",
        llm: "LLMInterface | None" = None,
        enable_hybrid: bool = True,
    ) -> None:
        """Initializes the search agent.

        Args:
            db: Vector database for search operations.
            llm: Optional LLM interface for query expansion. If not provided,
                the default LLM from factory will be used.
            enable_hybrid: Enable hybrid BM25+vector search with RRF fusion.
        """
        self.db = db
        self.llm = llm or LLMFactory.create_llm()
        self._hybrid_searcher = None
        self._enable_hybrid = enable_hybrid

    @property
    def hybrid_searcher(self):
        """Lazy-load HybridSearcher for BM25+vector fusion."""
        if self._hybrid_searcher is None and self._enable_hybrid:
            try:
                from core.retrieval.hybrid import HybridSearcher

                self._hybrid_searcher = HybridSearcher(self.db)
                log("[Search] HybridSearcher initialized")
            except Exception as e:
                log(f"[Search] HybridSearcher init failed: {e}")
        return self._hybrid_searcher

    @observe("search_parse_query")
    async def parse_query(self, query: str) -> ParsedQuery:
        """Parses and expands a natural language search query using an LLM.

        Args:
            query: The user's natural language search query.

        Returns:
            A ParsedQuery object containing structured filters and expanded text.
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
        """Resolves a person's name to a face cluster ID.

        Performs exact and fuzzy matching against the HITL identity database.

        Args:
            person_name: The name of the person to resolve.

        Returns:
            The cluster ID if found, otherwise None.
        """
        try:
            # 1. Try exact match first (fastest)
            cluster_id = self.db.get_cluster_id_by_name(person_name)
            if cluster_id is not None:
                return cluster_id

            # 2. Fallback to fuzzy match (handles typos, partial names)
            cluster_id = self.db.fuzzy_get_cluster_id_by_name(person_name)
            if cluster_id:
                log(
                    f"[Search] Fuzzy matched '{person_name}' → cluster {cluster_id}"
                )
            return cluster_id
        except Exception:
            return None

    def _get_face_ids_for_cluster(self, cluster_id: int) -> list[str]:
        """Retrieves all face point IDs belonging to a specific cluster.

        Args:
            cluster_id: The ID of the face cluster.

        Returns:
            A list of face point IDs (strings).
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
        """Performs a production-grade scene-level search with LLM expansion.

        Resolves identities, builds comprehensive filters for clothing,
        location, and actions, and queries the scene collection. Falls back
        to frame-level search if scene search fails.

        Args:
            query: The natural language search query.
            limit: Maximum number of results to return.
            use_expansion: Whether to use LLM query expansion.
            video_path: Optional filter for a specific video path.

        Returns:
            A dictionary containing results, parsed query, and metadata.
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
                log(
                    f"[Search] Resolved '{parsed.person_name}' → cluster {cluster_id}"
                )

        # 3. Build search query from expanded keywords
        search_text = parsed.to_search_text()
        log(f"[Search] Expanded search text: '{search_text}'")

        # 4. Execute scene search with comprehensive filters
        try:
            results = self.db.search_scenes(
                query=search_text,
                limit=limit,
                person_name=resolved_name,
                face_cluster_ids=[cluster_id]
                if cluster_id is not None
                else None,
                clothing_color=parsed.clothing_color,
                clothing_type=parsed.clothing_type,
                accessories=parsed.accessories if parsed.accessories else None,
                location=parsed.location,
                visible_text=parsed.text_to_find
                if parsed.text_to_find
                else None,
                action_keywords=parsed.action_keywords
                if parsed.action_keywords
                else None,
                video_path=video_path,
                search_mode="hybrid",
            )
            log(f"[Search] Found {len(results)} scene results")
        except Exception as e:
            log(
                f"[Search] Scene search failed: {e}, falling back to frame search"
            )
            results = await self._fallback_frame_search(
                parsed, search_text, limit
            )

        # GOLD.md Compliance: Comprehensive Search Reasoning Trace
        reasoning_chain = {
            "step1_parse": f"Parsed '{query[:50]}...' into structured query",
            "step2_identity": f"Resolved identity: {resolved_name or 'None'} → cluster {cluster_id}",
            "step3_expand": f"Expanded to: {search_text[:80]}...",
            "step4_filters": {
                "face_cluster": cluster_id,
                "clothing_color": parsed.clothing_color,
                "clothing_type": parsed.clothing_type,
                "accessories": parsed.accessories,
                "location": parsed.location,
                "visible_text": parsed.text_to_find,
                "video_path": video_path,
            },
            "step5_results": f"Found {len(results)} scenes",
        }
        top_scores = [
            {r.get("id", "?"): round(r.get("score", 0), 3)} for r in results[:5]
        ]
        log(f"[Search] Original: {query}")
        log(f"[Search] Expanded: {search_text}")
        log(
            f"[Search] Filters: faces={[cluster_id] if cluster_id else []}, "
            f"video={video_path or 'all'}"
        )
        log(f"[Search] Scoring: {top_scores}")
        log(f"[Search] Reasoning: {reasoning_chain}")

        return {
            "query": query,
            "parsed": parsed.model_dump(),
            "resolved_identity": resolved_name,
            "face_ids_matched": len(face_ids),
            "expanded_search": search_text,
            "results": results,
            "result_count": len(results),
            "search_type": "scene",
            "reasoning_chain": reasoning_chain,
        }

    @observe("search_agentic")
    async def search(
        self,
        query: str,
        limit: int = 20,
        use_expansion: bool = True,
    ) -> dict[str, Any]:
        """Performs a frame-level agentic search with query expansion.

        This is a legacy search method. For production use, use `search_scenes`.

        Args:
            query: The natural language search query.
            limit: Maximum number of results to return.
            use_expansion: Whether to use LLM query expansion.

        Returns:
            A dictionary containing results, parsed query, and metadata.
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
                log(
                    f"[Search] Resolved '{parsed.person_name}' → cluster {cluster_id} ({len(face_ids)} faces)"
                )

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
            log(
                f"[Search] Added clothing color filter: {parsed.clothing_color}"
            )

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
            specific_entities = [
                k for k in parsed.visual_keywords if len(k) > 3
            ]
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
            query_vector = self.db.encode_texts(
                search_text or "scene activity", is_query=True
            )[0]

            if filters:
                conditions: list[models.Condition] = list(filters)
                results = self.db.client.query_points(
                    collection_name=self.db.MEDIA_COLLECTION,
                    query=query_vector,
                    query_filter=models.Filter(should=conditions)
                    if len(conditions) > 1
                    else models.Filter(must=conditions),
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
        """Falls back to frame-level search if scene search is unavailable.

        Args:
            parsed: The already parsed query object.
            search_text: The expanded search text.
            limit: Maximum number of results to return.

        Returns:
            A list of search result dictionaries.
        """
        try:
            return self.db.search_frames(query=search_text, limit=limit)
        except Exception:
            return []

    async def search_simple(self, query: str, limit: int = 20) -> list[dict]:
        """Performs a simple frame search without LLM expansion.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            A list of search result dictionaries.
        """
        return self.db.search_frames(query=query, limit=limit)

    async def hybrid_search(
        self,
        query: str,
        limit: int = 50,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        video_id: str | None = None,
    ) -> list[dict]:
        """Perform hybrid BM25+vector search with weighted RRF fusion.

        Per AGENTS.MD: Use weighted RRF (0.7 vector, 0.3 BM25).

        Args:
            query: Search query string.
            limit: Maximum results to return.
            vector_weight: Weight for vector search (default 0.7).
            keyword_weight: Weight for BM25 keyword search (default 0.3).
            video_id: Optional filter by video.

        Returns:
            List of results with fused scores.
        """
        if self.hybrid_searcher:
            try:
                results = await self.hybrid_searcher.search(
                    query=query,
                    limit=limit,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                    video_id=video_id,
                )
                log(
                    f"[Search] Hybrid search: {len(results)} results "
                    f"(weights: {vector_weight:.1f}v/{keyword_weight:.1f}kw)"
                )
                return results
            except Exception as e:
                log(f"[Search] Hybrid search failed, falling back: {e}")

        # Fallback to vector-only search
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
        """Re-ranks search candidates using an LLM to verify all query constraints.

        Performs a second-stage verification where an LLM checks if the
        candidate segments actually satisfy the specific entities, actions,
        and attributes mentioned in the query.

        Args:
            query: The original user search query.
            candidates: A list of candidate search results.
            top_k: The number of top results to return after re-ranking.

        Returns:
            The re-ranked list of results with LLM verification metadata.
        """
        if not candidates:
            return []

        # Load rerank prompt from external file
        rerank_prompt = load_prompt("rerank_verification")

        reranked = []

        for candidate in candidates[
            : top_k * 2
        ]:  # Check more candidates than needed
            description = (
                candidate.get("description", "")
                or candidate.get("dense_caption", "")
                or candidate.get("raw_description", "")
            )

            prompt = rerank_prompt.format(
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
                    constraints_checked: list[dict] = Field(
                        default_factory=list
                    )
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
                    candidate.get("score", 0) * 0.4  # Vector similarity
                    + result.match_score * 0.6  # LLM verification
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
        """Performs a state-of-the-art search with full verification and explainability.

        This pipeline combines dynamic query expansion, multi-vector hybrid
        search, and LLM-based re-ranking to provide highly accurate and
        transparent search results.

        Args:
            query: The complex natural language query.
            limit: Maximum number of results to return.
            video_path: Optional filter for a specific video.
            use_reranking: Whether to perform second-stage LLM verification.

        Returns:
            A dictionary containing ranked results and reasoning for each match.
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
        if hasattr(parsed, "entities") and parsed.entities:
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
        # Use fallback chain: scenes → scenelets → frames
        fallback_used = None
        try:
            # Use search_scenes as the explainable/SOTA method
            candidates = self.db.search_scenes(
                query=search_text,
                limit=limit * 3 if use_reranking else limit,
                person_name=person_names[0] if person_names else None,
                face_cluster_ids=face_ids if face_ids else None,
                clothing_color=parsed.clothing_color
                if hasattr(parsed, "clothing_color")
                else None,
                clothing_type=parsed.clothing_type
                if hasattr(parsed, "clothing_type")
                else None,
                accessories=parsed.accessories
                if hasattr(parsed, "accessories")
                else None,
                location=parsed.location
                if hasattr(parsed, "location")
                else None,
                visible_text=parsed.text_to_find
                if hasattr(parsed, "text_to_find")
                else None,
                action_keywords=parsed.action_keywords
                if hasattr(parsed, "action_keywords")
                else None,
                video_path=video_path,
                search_mode="hybrid",
            )
            log(
                f"[SOTA Search] Retrieved {len(candidates)} candidates from scenes"
            )

            # Fallback 1: Try scenelets if scenes empty
            if not candidates:
                log("[SOTA Search] No scenes found, falling back to scenelets")
                fallback_used = "scenelets"
                try:
                    candidates = self.db.search_scenelets(
                        query=search_text,
                        limit=limit * 3 if use_reranking else limit,
                        video_path=video_path,
                    )
                    log(
                        f"[SOTA Search] Retrieved {len(candidates)} candidates from scenelets"
                    )
                except Exception as e:
                    log(f"[SOTA Search] Scenelet search failed: {e}")
                    candidates = []

            # Fallback 2: Try frame-level hybrid search if still empty
            if not candidates:
                log("[SOTA Search] No scenelets found, falling back to frames")
                fallback_used = "frames"
                face_cluster_id = face_ids[0] if face_ids else None
                candidates = self.db.search_frames_hybrid(
                    query=search_text,
                    limit=limit * 3 if use_reranking else limit,
                    face_cluster_ids=[face_cluster_id]
                    if face_cluster_id is not None
                    else None,
                )
                log(
                    f"[SOTA Search] Retrieved {len(candidates)} candidates from frames"
                )

        except Exception as e:
            log(f"[SOTA Search] Search failed: {e}")
            candidates = []

        # 4. Re-rank with LLM verification (optional but recommended)
        if use_reranking and candidates:
            try:
                candidates = await self.rerank_with_llm(
                    query, candidates, top_k=limit
                )
                log(f"[SOTA Search] Re-ranked to {len(candidates)} results")
            except Exception as e:
                log(f"[SOTA Search] Re-ranking failed: {e}, using raw results")

        # GOLD.md Compliance: Comprehensive Search Reasoning Trace with Pipeline Steps
        # Build detailed pipeline_steps for frontend debug panel
        pipeline_steps = [
            {
                "step": "Query Parsing",
                "status": "completed",
                "detail": f"Extracted {len(parsed.entities) if hasattr(parsed, 'entities') and parsed.entities else 0} entities",
                "data": {
                    "original_query": query[:100],
                    "entities_found": [e.name for e in parsed.entities]
                    if hasattr(parsed, "entities") and parsed.entities
                    else [],
                    "scene_description": parsed.scene_description[:100]
                    if hasattr(parsed, "scene_description")
                    and parsed.scene_description
                    else None,
                },
            },
            {
                "step": "Identity Resolution",
                "status": "completed",
                "detail": f"Matched {len(person_names)} names → {len(face_ids)} faces",
                "data": {
                    "names_searched": person_names,
                    "faces_matched": len(face_ids),
                },
            },
            {
                "step": "Query Expansion",
                "status": "completed",
                "detail": f"Expanded to {len(search_text)} chars",
                "data": {
                    "expanded_text": search_text[:200] + "..."
                    if len(search_text) > 200
                    else search_text,
                },
            },
            {
                "step": "Vector Search",
                "status": "completed",
                "detail": f"{'Fallback: ' + fallback_used if fallback_used else 'scenes'} → {len(candidates)} results",
                "data": {
                    "collection_searched": fallback_used or "scenes",
                    "fallback_used": fallback_used is not None,
                    "candidates_found": len(candidates),
                },
            },
            {
                "step": "LLM Reranking",
                "status": "completed" if use_reranking else "skipped",
                "detail": f"{'Applied' if use_reranking else 'Disabled'} → {len(candidates[:limit])} final results",
                "data": {
                    "enabled": use_reranking,
                    "final_count": len(candidates[:limit]),
                },
            },
        ]

        reasoning_chain = {
            "step1_parse": f"Parsed '{query[:50]}...' → dynamic entities",
            "step2_identity": f"Resolved: {person_names} → {len(face_ids)} faces",
            "step3_expand": f"Search text: {search_text[:80]}...",
            "step4_retrieve": f"Retrieved {len(candidates)} candidates from {fallback_used or 'scenes'}",
            "step5_rerank": f"Reranking={use_reranking}, final={len(candidates[:limit])}",
        }
        log(f"[SOTA Search] Reasoning: {reasoning_chain}")

        # 5. Build response with full explainability
        return {
            "query": query,
            "parsed": parsed.model_dump()
            if hasattr(parsed, "model_dump")
            else {},
            "search_text": search_text,
            "person_names_resolved": person_names,
            "face_ids_matched": len(face_ids),
            "results": candidates[:limit],
            "result_count": len(candidates[:limit]),
            "search_type": "sota",
            "reranking_used": use_reranking,
            "fallback_used": fallback_used,
            "reasoning_chain": reasoning_chain,
            "pipeline_steps": pipeline_steps,
        }

    @observe("scenelet_search")
    async def scenelet_search(
        self,
        query: str,
        video_path: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Searches for action-based video segments with temporal context.

        Identifies relevant moments and expands them into short windows
        (scenelets) to capture the full context of an action or event.

        Args:
            query: The natural language search query.
            video_path: Optional filter for a specific video.
            limit: Maximum number of results to return.

        Returns:
            A dictionary with results containing start and end time ranges.
        """
        log(f"[Scenelet Search] Query: '{query[:80]}...'")

        parsed = await self.parse_query(query)
        search_text = parsed.to_search_text() or query

        try:
            # Use dedicated Scenelet search from DB
            results = self.db.search_scenelets(
                query=search_text,
                limit=limit,
                video_path=video_path,
            )

            formatted_results = []
            for r in results:
                # Format for API consistency
                start = r.get("start_time", 0.0)
                end = r.get("end_time", 0.0)
                text = r.get("text", "")

                formatted_results.append(
                    {
                        **r,
                        "reasoning": f"Matched scenelet ({start:.1f}s-{end:.1f}s): {text[:100]}...",
                        "match_explanation": f"Action sequence detected: {text[:50]}",
                    }
                )

            return {
                "query": query,
                "search_type": "scenelet",
                "results": formatted_results,
                "result_count": len(formatted_results),
            }

        except AttributeError:
            # Fallback if DB method not ready (during migration)
            log(
                "[Scenelet Search] DB method not found, falling back to frame simulation"
            )
            candidates = self.db.search_frames(
                query=search_text, limit=limit * 2
            )

            results_with_ranges = []
            for cand in candidates:
                ts = cand.get("timestamp", 0)
                start_time = max(0, ts - 2.5)
                end_time = ts + 2.5

                actions = cand.get("actions", [])
                entities = cand.get("entities", cand.get("entity_names", []))

                action_str = ", ".join(actions[:3]) if actions else "activity"
                entity_str = ", ".join(entities[:3]) if entities else "scene"

                reasoning = f"Matched Sequence: {action_str} with {entity_str} from {start_time:.1f}s to {end_time:.1f}s"

                results_with_ranges.append(
                    {
                        **cand,
                        "start_time": start_time,
                        "end_time": end_time,
                        "reasoning": reasoning,
                        "match_explanation": f"Action '{action_str}' detected at {ts:.1f}s",
                    }
                )

            return {
                "query": query,
                "search_type": "scenelet",
                "results": results_with_ranges[:limit],
                "result_count": len(results_with_ranges[:limit]),
            }
